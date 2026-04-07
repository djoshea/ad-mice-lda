function features = compute_features(cfg, trials)
% COMPUTE_FEATURES  CV partitions, feature timebase, PSD + envelope extraction, cumsum.
%
%   features = compute_features(cfg, trials)
%
% Checks for cached artifact at <outdir>/features_<mousename>.mat.

features = FeatureData();

% --- cache check ---
matfile = fullfile(cfg.outdir, sprintf('features_%s.mat', cfg.mousename));
if exist(matfile, 'file')
    tmp = load(matfile);
    if isfield(tmp, 'saved_cfg') && isequal(tmp.saved_cfg, cfg)
        flds = fieldnames(tmp.features);
        for fi = 1:numel(flds)
            features.(flds{fi}) = tmp.features.(flds{fi});
        end
        fprintf('Loaded cached features_%s.mat\n', cfg.mousename);
        return;
    end
end

% --- unpack for readability ---
fs = cfg.fs;
ntp = trials.ntp;
trialidx = trials.trialidx;
cut_samp = trials.cut_samp;
npresamp = trials.npresamp;
triallensamp = trials.triallensamp;
puff_cut = trials.puff_cut;

% --- cv partitions, matched repeated cv, trial-grouped ---
rng(cfg.random_state);

ktrial = min(cfg.n_splits, ntp);
if ktrial < 2
    error('Not enough TP trials for CV. Increase nTP or reduce N_SPLITS.');
end

cv_repeats = cfg.n_cv_repeats;
splitsrep = cell(cv_repeats, 1);
for rr = 1:cv_repeats
    rng(cfg.random_state + 1000*rr, 'twister');

    cvt = cvpartition(ntp, 'KFold', ktrial);
    splits = cell(cvt.NumTestSets, 1);

    for kk = 1:cvt.NumTestSets
        tetrials = find(test(cvt, kk));
        trtrials = find(training(cvt, kk));

        teidx = [tetrials; tetrials + ntp];
        tridx = [trtrials; trtrials + ntp];

        splits{kk} = struct('train', tridx, 'test', teidx);
    end

    splitsrep{rr} = splits;
end

fprintf('CV: trial-grouped K=%d folds, R=%d repeats (matched for OBS + PERMS)\n', ktrial, cv_repeats);

% --- config-specific setup ---
usetargets = [1 2];

usealphapsd   = contains(cfg.bandsstr, "alphaPSD", "IgnoreCase", true);
uselowbetapsd = contains(cfg.bandsstr, "lowbetaPSD", "IgnoreCase", true);
usedeltaenv   = contains(cfg.bandsstr, "deltaENV", "IgnoreCase", true);
usethetaenv   = contains(cfg.bandsstr, "thetaENV", "IgnoreCase", true);

if ~(usealphapsd || uselowbetapsd || usedeltaenv || usethetaenv)
    error('bandsUsed has no recognized features: %s', cfg.bandsstr);
end

featorder = {};
if usealphapsd,   featorder{end+1} = 'alphaPSD';   end %#ok<AGROW>
if uselowbetapsd, featorder{end+1} = 'lowbetaPSD'; end %#ok<AGROW>
if usedeltaenv,   featorder{end+1} = 'deltaENV';   end %#ok<AGROW>
if usethetaenv,   featorder{end+1} = 'thetaENV';   end %#ok<AGROW>

nfeatperc = numel(featorder);
nc = numel(usetargets);
nfeat = nfeatperc * nc;

% --- feature timebase ---
wins = round(cfg.win_ms_feat * fs / 1000);
steps = round(cfg.step_ms_psd * fs / 1000);

starts_full = 1:steps:(triallensamp - wins + 1);
times_trial_full = ((starts_full + (wins-1)/2) - (npresamp+1)) / fs;

% hard pre-puff cutoff for feature bin centers
halfwin_feat_s = (wins - 1) / (2 * fs);
tmax_feat_center = puff_cut - halfwin_feat_s - 1e-12;

keep_feat = (times_trial_full <= tmax_feat_center);
starts = starts_full(keep_feat);
times_trial = times_trial_full(keep_feat);

ntimebins = numel(starts);
if ntimebins < 2
    error('Too few feature bins after hard pre-puff cutoff.');
end

basemask = (times_trial >= cfg.base_start_default) & (times_trial <= cfg.base_end_default);
if ~any(basemask)
    error('No baseline bins in [%.2f %.2f].', cfg.base_start_default, cfg.base_end_default);
end

if numel(times_trial) < 2
    error('times_trial <2 bins.');
end

dt_bin = times_trial(2) - times_trial(1);
if dt_bin <= 0 || ~isfinite(dt_bin)
    error('Bad dt_bin.');
end
t0_bin = times_trial(1);
nbins = numel(times_trial);

% --- prep filters ---
if usedeltaenv
    wnd = cfg.delta_band / (fs/2);
    [bd, ad] = butter(cfg.env_order, wnd, 'bandpass');
else
    bd = []; ad = [];
end

if usethetaenv
    wnt = cfg.theta_band / (fs/2);
    if any(wnt <= 0) || any(wnt >= 1)
        error('Bad THETA_BAND relative to Fs.');
    end
    [bt, at] = butter(cfg.env_order, wnt, 'bandpass');
else
    bt = []; at = [];
end

% --- compute features per contact ---
featmat = zeros(ntp, ntimebins, nfeat);
featnames = strings(1, nfeat);
nametarget = trials.nametarget;
sig_bytarget = trials.sig_bytarget;
nw = cfg.nw;
fmax = cfg.fmax;
alpha_band = cfg.alpha_band;
lowbeta_band = cfg.lowbeta_band;
envmetric = cfg.envmetric;

for cc = 1:nc

    sig_cont = sig_bytarget{usetargets(cc)};

    psd_alpha = zeros(ntp, ntimebins);
    psd_lowb  = zeros(ntp, ntimebins);
    env_delta = zeros(ntp, ntimebins);
    env_theta = zeros(ntp, ntimebins);

    % psd (alpha/lowbeta)
    if usealphapsd || uselowbetapsd

        freqs = []; idx_alpha = []; idx_lowb = [];

        for tr = 1:ntp
            seg = sig_cont(trialidx(tr, :));
            seg = seg(1:min(numel(seg), cut_samp));
            seg = seg(1:min(numel(seg), cut_samp));
            m = [];

            for ti = 1:ntimebins
                s0 = starts(ti);
                winseg = seg(s0:(s0+wins-1));
                [pxx, f] = pmtm(winseg, nw, [], fs);

                if ~isempty(fmax)
                    keepf = (f <= fmax);
                    pxx = pxx(keepf);
                    f = f(keepf);
                end

                if isempty(freqs)
                    freqs = f(:)';
                    if usealphapsd
                        idx_alpha = find(freqs >= alpha_band(1) & freqs < alpha_band(2));
                        if isempty(idx_alpha)
                            error('No alpha freqs in [%g %g] under FMAX=%g.', alpha_band(1), alpha_band(2), fmax);
                        end
                    end
                    if uselowbetapsd
                        idx_lowb = find(freqs >= lowbeta_band(1) & freqs < lowbeta_band(2));
                        if isempty(idx_lowb)
                            error('No lowbeta freqs in [%g %g] under FMAX=%g.', lowbeta_band(1), lowbeta_band(2), fmax);
                        end
                    end
                end

                if isempty(m)
                    m = zeros(numel(freqs), ntimebins);
                end
                m(:, ti) = pxx(:);
            end

            mu = mean(m(:, basemask), 2);
            sd = std(m(:, basemask), 0, 2);
            sd(sd < 1e-12) = 1;
            mz = (m - mu) ./ sd;

            if usealphapsd
                psd_alpha(tr, :) = mean(mz(idx_alpha, :), 1);
            end
            if uselowbetapsd
                psd_lowb(tr, :) = mean(mz(idx_lowb, :), 1);
            end
        end

    end

    % env delta/theta, causal filter
    if usedeltaenv || usethetaenv

        for tr = 1:ntp
            seg = sig_cont(trialidx(tr, :));
            seg = seg(1:min(numel(seg), cut_samp));
            seg = seg(1:min(numel(seg), cut_samp));

            if usedeltaenv
                xd = filter(bd, ad, double(seg));
                if envmetric == "rect"
                    aenv = abs(xd);
                    for ti = 1:ntimebins
                        s0 = starts(ti);
                        idxw = s0:(s0+wins-1);
                        env_delta(tr, ti) = mean(aenv(idxw));
                    end
                elseif envmetric == "rms"
                    x2 = xd.^2;
                    for ti = 1:ntimebins
                        s0 = starts(ti);
                        idxw = s0:(s0+wins-1);
                        env_delta(tr, ti) = sqrt(mean(x2(idxw)));
                    end
                else
                    error('Unknown envMetric: %s', envmetric);
                end

                mu = mean(env_delta(tr, basemask));
                sd = std(env_delta(tr, basemask), 0, 2);
                if sd < 1e-12, sd = 1; end
                env_delta(tr, :) = (env_delta(tr, :) - mu) ./ sd;
            end

            if usethetaenv
                xt = filter(bt, at, double(seg));
                if envmetric == "rect"
                    aenv = abs(xt);
                    for ti = 1:ntimebins
                        s0 = starts(ti);
                        idxw = s0:(s0+wins-1);
                        env_theta(tr, ti) = mean(aenv(idxw));
                    end
                elseif envmetric == "rms"
                    x2 = xt.^2;
                    for ti = 1:ntimebins
                        s0 = starts(ti);
                        idxw = s0:(s0+wins-1);
                        env_theta(tr, ti) = sqrt(mean(x2(idxw)));
                    end
                else
                    error('Unknown envMetric: %s', envmetric);
                end

                mu = mean(env_theta(tr, basemask));
                sd = std(env_theta(tr, basemask), 0, 2);
                if sd < 1e-12, sd = 1; end
                env_theta(tr, :) = (env_theta(tr, :) - mu) ./ sd;
            end

        end

    end

    % pack into featmat
    for ff = 1:nfeatperc
        fname = featorder{ff};
        idxglobal = (cc-1)*nfeatperc + ff;
        featnames(idxglobal) = string(sprintf('%s_%s', nametarget{usetargets(cc)}, fname));

        if strcmpi(fname, 'alphaPSD')
            featmat(:, :, idxglobal) = psd_alpha;
        elseif strcmpi(fname, 'lowbetaPSD')
            featmat(:, :, idxglobal) = psd_lowb;
        elseif strcmpi(fname, 'deltaENV')
            featmat(:, :, idxglobal) = env_delta;
        elseif strcmpi(fname, 'thetaENV')
            featmat(:, :, idxglobal) = env_theta;
        else
            error('Unknown feature name in featOrder: %s', fname);
        end
    end

end

% --- cumsums for fast window means ---
cs_all = zeros(ntp, ntimebins+1, nfeat);
for ff = 1:nfeat
    cs_all(:, :, ff) = [zeros(ntp, 1), cumsum(featmat(:, :, ff), 2)];
end

% --- populate output ---
features.featmat    = featmat;
features.featnames  = featnames;
features.featorder  = featorder;
features.cs_all     = cs_all;
features.splitsrep  = splitsrep;
features.ktrial     = ktrial;
features.times_trial = times_trial;
features.basemask   = basemask;
features.starts     = starts;
features.ntimebins  = ntimebins;
features.dt_bin     = dt_bin;
features.t0_bin     = t0_bin;
features.nbins      = nbins;
features.wins       = wins;
features.nfeat      = nfeat;
features.nfeatperc  = nfeatperc;
features.nc         = nc;

% --- save artifact ---
if ~exist(cfg.outdir, 'dir'), mkdir(cfg.outdir); end
saved_cfg = cfg; %#ok<NASGU>
save(matfile, 'features', 'saved_cfg', '-v7.3');
fprintf('Saved features_%s.mat\n', cfg.mousename);

end
