function results = run_decoding(cfg, trials, features)
% RUN_DECODING  Define decoding grid, run observed LDA + permutation null.
%
%   results = run_decoding(cfg, trials, features)
%
% Checks for cached artifact at <outdir>/decoding_<mousename>.mat.

results = DecodingResults();

% --- cache check ---
matfile = fullfile(cfg.outdir, sprintf('decoding_%s.mat', cfg.mousename));
if exist(matfile, 'file')
    tmp = load(matfile);
    if isfield(tmp, 'saved_cfg') && isequal(tmp.saved_cfg, cfg)
        flds = fieldnames(tmp.results);
        for fi = 1:numel(flds)
            results.(flds{fi}) = tmp.results.(flds{fi});
        end
        fprintf('Loaded cached decoding_%s.mat\n', cfg.mousename);
        return;
    end
end

tstartdec = tic;

% --- unpack for readability and parfor transparency ---
fs          = cfg.fs;
ntp         = trials.ntp;
puff_cut    = trials.puff_cut;
puff_rel    = trials.puff_rel;
rowstp      = trials.rowstp;
cs_all      = features.cs_all;
wins        = features.wins;
t0_bin      = features.t0_bin;
dt_bin      = features.dt_bin;
nbins       = features.nbins;
nfeat       = features.nfeat;
splitsrep   = features.splitsrep;
times_trial = features.times_trial;

gammashrink   = cfg.gammashrink;
random_state  = cfg.random_state;
cv_repeats    = cfg.n_cv_repeats;
npermsetting  = cfg.nperm;

% --- decoding window grid ---
win_dec_s  = cfg.win_ms_dec / 1000;
step_dec_s = cfg.step_ms_dec / 1000;

starts_dec = cfg.analysis_t1:step_dec_s:(cfg.analysis_t2 - win_dec_s + 1e-12);
nwin = numel(starts_dec);
if nwin < 2
    error('Too few decoding windows.');
end

winbounds    = zeros(nwin, 2);
times_center = zeros(nwin, 1);
times_end    = zeros(nwin, 1);
for wi = 1:nwin
    winbounds(wi, :) = [starts_dec(wi), starts_dec(wi) + win_dec_s];
    times_center(wi) = starts_dec(wi) + win_dec_s/2;
    times_end(wi)    = starts_dec(wi) + win_dec_s;
end

% --- hard pre-puff restriction ---
halfwin_feat_s = (wins - 1) / (2 * fs);
keep_win = (times_end <= (puff_cut - halfwin_feat_s + 1e-12));

starts_dec   = starts_dec(keep_win);
winbounds    = winbounds(keep_win, :);
times_center = times_center(keep_win);
times_end    = times_end(keep_win);
nwin         = numel(starts_dec);

if nwin < 2
    error('too few pre-puff decoding windows after masking (nwin=%d).', nwin);
end

% --- pseudo-anchor range constraints ---
anchorl = cfg.base_start_default - cfg.analysis_t1;
anchoru = cfg.base_end_default   - cfg.analysis_t2;

tminavail = min(times_trial);
tmaxavail = max(times_trial);
anchorl = max(anchorl, tminavail - cfg.analysis_t1);
anchoru = min(anchoru, tmaxavail - cfg.analysis_t2);

if anchoru <= anchorl
    error('No valid pseudo-anchor range after constraints.');
end

% --- precompute tone-locked bin ranges ---
i0tone = zeros(nwin, 1);
i1tone = zeros(nwin, 1);
for wi = 1:nwin
    w0 = winbounds(wi, 1); w1 = winbounds(wi, 2);
    i0 = round((w0 - t0_bin) / dt_bin) + 1;
    i1 = round((w1 - t0_bin) / dt_bin) + 1;
    i0 = max(1, min(nbins, i0));
    i1 = max(1, min(nbins, i1));
    if i1 < i0, i1 = i0; end
    i0tone(wi) = i0;
    i1tone(wi) = i1;
end

% --- build observed x_rand vs x_tone ---
rng(random_state);
anchors_obs = anchorl + (anchoru - anchorl) .* rand(ntp, 1);

x_tone = zeros(ntp, nwin, nfeat);
x_rand = zeros(ntp, nwin, nfeat);

for wi = 1:nwin
    i0t = i0tone(wi); i1t = i1tone(wi);
    lent = max(1, (i1t - i0t + 1));

    wrel0 = winbounds(wi, 1);
    wrel1 = winbounds(wi, 2);

    a = anchors_obs;
    w0_r = a + wrel0; w1_r = a + wrel1;
    i0 = round((w0_r - t0_bin) / dt_bin) + 1;
    i1 = round((w1_r - t0_bin) / dt_bin) + 1;
    i0 = max(1, min(nbins, i0));
    i1 = max(1, min(nbins, i1));
    bad = (i1 < i0); i1(bad) = i0(bad);

    i0c = max(1, min(nbins, i0));
    i1c = max(1, min(nbins, i1));
    bad2 = (i1c < i0c); i1c(bad2) = i0c(bad2);
    len = max(1, (i1c - i0c + 1));

    for ff = 1:nfeat
        csf = cs_all(:, :, ff);
        x_tone(:, wi, ff) = (csf(:, i1t+1) - csf(:, i0t)) ./ lent;
        idx1 = sub2ind(size(csf), rowstp, i1c+1);
        idx0 = sub2ind(size(csf), rowstp, i0c);
        s = csf(idx1) - csf(idx0);
        x_rand(:, wi, ff) = s ./ len;
    end
end

xobs = cat(1, x_rand, x_tone);
y0 = [zeros(ntp, 1); ones(ntp, 1)];

% --- observed bacc across repeats ---
bacc_rep = nan(cv_repeats, nwin);

for rr = 1:cv_repeats
    splits = splitsrep{rr};

    for wi = 1:nwin
        xw = squeeze(xobs(:, wi, :));
        yhat_all = nan(size(y0));

        for kk = 1:numel(splits)
            tridx = splits{kk}.train;
            teidx = splits{kk}.test;

            xtr = xw(tridx, :); ytr = y0(tridx);

            mu = mean(xtr, 1);
            sd = std(xtr, 0, 1); sd(sd < 1e-12) = 1;

            xtrz = (xtr - mu) ./ sd;
            xtez = (xw(teidx, :) - mu) ./ sd;

            try
                mdl = fitcdiscr(xtrz, ytr, 'DiscrimType', 'linear', 'Gamma', gammashrink);
                yhat = predict(mdl, xtez);
            catch
                yhat = mode(ytr) * ones(numel(teidx), 1);
            end

            yhat_all(teidx) = yhat;
        end

        tpr = mean(yhat_all(y0==1) == 1);
        tnr = mean(yhat_all(y0==0) == 0);
        bacc_rep(rr, wi) = 0.5 * (tpr + tnr);
    end

    if mod(rr, max(1, round(cv_repeats/5))) == 0 || rr == 1 || rr == cv_repeats
        fprintf('  CV repeat %d/%d\n', rr, cv_repeats);
        toc
    end
end

bacc_obs   = mean(bacc_rep, 1, 'omitnan')';
bacc_ci_lo = prctile(bacc_rep, 2.5, 1)';
bacc_ci_hi = prctile(bacc_rep, 97.5, 1)';

% --- permutation null ---
null_bacc = nan(npermsetting, nwin);

usepar = (~isempty(gcp('nocreate'))) && (npermsetting >= 250);
if usepar
    parfor pi = 1:npermsetting
        rng(random_state + 10000 + pi, 'twister');

        anchors_pi = anchorl + (anchoru - anchorl) .* rand(ntp, 1);
        x_rand_pi = zeros(ntp, nwin, nfeat);

        for wi = 1:nwin
            wrel0 = winbounds(wi, 1);
            wrel1 = winbounds(wi, 2);

            a = anchors_pi;
            w0_r = a + wrel0; w1_r = a + wrel1;
            i0 = round((w0_r - t0_bin) / dt_bin) + 1;
            i1 = round((w1_r - t0_bin) / dt_bin) + 1;
            i0 = max(1, min(nbins, i0));
            i1 = max(1, min(nbins, i1));
            bad = (i1 < i0); i1(bad) = i0(bad);

            i0c = max(1, min(nbins, i0));
            i1c = max(1, min(nbins, i1));
            bad2 = (i1c < i0c); i1c(bad2) = i0c(bad2);
            len = max(1, (i1c - i0c + 1));

            for ff = 1:nfeat
                csf = cs_all(:, :, ff);
                idx1 = sub2ind(size(csf), rowstp, i1c+1);
                idx0 = sub2ind(size(csf), rowstp, i0c);
                s = csf(idx1) - csf(idx0);
                x_rand_pi(:, wi, ff) = s ./ len;
            end
        end

        xpi = cat(1, x_rand_pi, x_tone);

        bacc_pi_rep = nan(cv_repeats, nwin);

        for rr = 1:cv_repeats
            splits = splitsrep{rr};

            for wi = 1:nwin
                xw = squeeze(xpi(:, wi, :));
                yhat_all = nan(size(y0));

                for kk = 1:numel(splits)
                    tridx = splits{kk}.train;
                    teidx = splits{kk}.test;

                    xtr = xw(tridx, :); ytr = y0(tridx);

                    mu = mean(xtr, 1);
                    sd = std(xtr, 0, 1); sd(sd < 1e-12) = 1;

                    xtrz = (xtr - mu) ./ sd;
                    xtez = (xw(teidx, :) - mu) ./ sd;

                    try
                        mdl = fitcdiscr(xtrz, ytr, 'DiscrimType', 'linear', 'Gamma', gammashrink);
                        yhat = predict(mdl, xtez);
                    catch
                        yhat = mode(ytr) * ones(numel(teidx), 1);
                    end

                    yhat_all(teidx) = yhat;
                end

                tpr = mean(yhat_all(y0==1) == 1);
                tnr = mean(yhat_all(y0==0) == 0);
                bacc_pi_rep(rr, wi) = 0.5 * (tpr + tnr);
            end
        end

        null_bacc(pi, :) = mean(bacc_pi_rep, 1, 'omitnan');
    end
else
    for pi = 1:npermsetting
        rng(random_state + 10000 + pi, 'twister');

        anchors_pi = anchorl + (anchoru - anchorl) .* rand(ntp, 1);
        x_rand_pi = zeros(ntp, nwin, nfeat);

        for wi = 1:nwin
            wrel0 = winbounds(wi, 1);
            wrel1 = winbounds(wi, 2);

            a = anchors_pi;
            w0_r = a + wrel0; w1_r = a + wrel1;
            i0 = round((w0_r - t0_bin) / dt_bin) + 1;
            i1 = round((w1_r - t0_bin) / dt_bin) + 1;
            i0 = max(1, min(nbins, i0));
            i1 = max(1, min(nbins, i1));
            bad = (i1 < i0); i1(bad) = i0(bad);

            i0c = max(1, min(nbins, i0));
            i1c = max(1, min(nbins, i1));
            bad2 = (i1c < i0c); i1c(bad2) = i0c(bad2);
            len = max(1, (i1c - i0c + 1));

            for ff = 1:nfeat
                csf = cs_all(:, :, ff);
                idx1 = sub2ind(size(csf), rowstp, i1c+1);
                idx0 = sub2ind(size(csf), rowstp, i0c);
                s = csf(idx1) - csf(idx0);
                x_rand_pi(:, wi, ff) = s ./ len;
            end
        end

        xpi = cat(1, x_rand_pi, x_tone);

        bacc_pi_rep = nan(cv_repeats, nwin);

        for rr = 1:cv_repeats
            splits = splitsrep{rr};

            for wi = 1:nwin
                xw = squeeze(xpi(:, wi, :));
                yhat_all = nan(size(y0));

                for kk = 1:numel(splits)
                    tridx = splits{kk}.train;
                    teidx = splits{kk}.test;

                    xtr = xw(tridx, :); ytr = y0(tridx);

                    mu = mean(xtr, 1);
                    sd = std(xtr, 0, 1); sd(sd < 1e-12) = 1;

                    xtrz = (xtr - mu) ./ sd;
                    xtez = (xw(teidx, :) - mu) ./ sd;

                    try
                        mdl = fitcdiscr(xtrz, ytr, 'DiscrimType', 'linear', 'Gamma', gammashrink);
                        yhat = predict(mdl, xtez);
                    catch
                        yhat = mode(ytr) * ones(numel(teidx), 1);
                    end

                    yhat_all(teidx) = yhat;
                end

                tpr = mean(yhat_all(y0==1) == 1);
                tnr = mean(yhat_all(y0==0) == 0);
                bacc_pi_rep(rr, wi) = 0.5 * (tpr + tnr);
            end
        end

        null_bacc(pi, :) = mean(bacc_pi_rep, 1, 'omitnan');

        if mod(pi, max(1, round(npermsetting/10))) == 0 || pi == 1 || pi == npermsetting
            fprintf('  perm %d/%d (elapsed %.1fs)\n', pi, npermsetting, toc(tstartdec));
        end
    end
end

% --- pre-puff mask ---
mpre = false(nwin, 1);
if isfinite(puff_rel)
    mpre = (times_end <= (puff_rel + 1e-12));
end

% --- populate output ---
results.bacc_obs     = bacc_obs;
results.bacc_ci_lo   = bacc_ci_lo;
results.bacc_ci_hi   = bacc_ci_hi;
results.null_bacc    = null_bacc;
results.times_end    = times_end;
results.times_center = times_center;
results.winbounds    = winbounds;
results.nwin         = nwin;
results.mpre         = mpre;
results.puff_rel     = puff_rel;
results.anchors_obs  = anchors_obs;
results.xobs         = xobs;
results.x_tone       = x_tone;
results.y0           = y0;

% --- save artifact ---
if ~exist(cfg.outdir, 'dir'), mkdir(cfg.outdir); end
saved_cfg = cfg; %#ok<NASGU>
save(matfile, 'results', 'saved_cfg', '-v7.3');
fprintf('Saved decoding_%s.mat\n', cfg.mousename);

end
