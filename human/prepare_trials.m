function trials = prepare_trials(cfg)
% PREPARE_TRIALS  Load CSV, define tone+puff trials, compute puff lag, resolve contacts.
%
%   trials = prepare_trials(cfg)
%
% Checks for cached artifact at <outdir>/trials_<mousename>.mat.

trials = TonePuffTrialData();

% --- cache check ---
matfile = fullfile(cfg.outdir, sprintf('trials_%s.mat', cfg.mousename));
if exist(matfile, 'file')
    tmp = load(matfile);
    if isfield(tmp, 'saved_cfg') && isequal(tmp.saved_cfg, cfg)
        flds = fieldnames(tmp.trials);
        for fi = 1:numel(flds)
            trials.(flds{fi}) = tmp.trials.(flds{fi});
        end
        fprintf('Loaded cached trials_%s.mat\n', cfg.mousename);
        return;
    end
end

% --- load csv ---
csvfile = char(cfg.csvfile);
if ~exist(csvfile, 'file')
    error('CSV not found: %s', csvfile);
end

traw = readtable(csvfile, 'VariableNamingRule', 'preserve');
vars = traw.Properties.VariableNames;

if ~ismember('time', vars),  error('CSV must contain a column named "time".');  end
if ~ismember('event', vars), error('CSV must contain a column named "event".'); end

% drop rain5 if present
israin5 = strcmpi(vars, 'RAIN5');
if any(israin5)
    traw(:, israin5) = [];
    vars = traw.Properties.VariableNames;
    fprintf('Removing contact column(s): RAIN5\n');
end

contactcolsorig = setdiff(vars, {'time','event'}, 'stable');
if isempty(contactcolsorig)
    error('No contact columns found besides time/event.');
end
contactcols = matlab.lang.makeValidName(contactcolsorig, 'ReplacementStyle', 'delete');

% --- time axis ---
tcol = traw.('time');
if isnumeric(tcol)
    t_raw_sec = double(tcol(:));
else
    ts = string(tcol);
    t_raw_sec = nan(numel(ts), 1);
    ncolons = count(ts, ":");

    idx2 = (ncolons == 2);
    if any(idx2)
        d = duration(ts(idx2), 'InputFormat', 'hh:mm:ss.S');
        t_raw_sec(idx2) = seconds(d);
    end

    idx1 = (ncolons == 1);
    if any(idx1)
        d = duration(ts(idx1), 'InputFormat', 'mm:ss.S');
        t_raw_sec(idx1) = seconds(d);
    end

    idxbad = ~isfinite(t_raw_sec);
    if any(idxbad)
        t_raw_sec(idxbad) = str2double(ts(idxbad));
    end

    if any(~isfinite(t_raw_sec))
        error('Failed to parse some entries in time column to seconds.');
    end
end

t_raw = double(t_raw_sec(:));
n = numel(t_raw);
if n < 1 || ~isfinite(t_raw(1))
    error('time column first entry is not finite; cannot anchor uniform timebase.');
end

fs = cfg.fs;
dt_fixed = 1 / fs;
t_sec = t_raw(1) + (0:(n-1))' * dt_fixed;
fprintf('FORCED Fs=%.6f Hz (dt=%.6g s). Time axis anchored at t_raw(1)=%.6f\n', fs, dt_fixed, t_raw(1));

% --- events ---
ev = traw.('event');
if iscell(ev)
    ev = cellfun(@(x) str2double(string(x)), ev);
end
ev = double(ev);
ev(~isfinite(ev)) = NaN;

toneendidx_all = find(ev == 1);
puffidx_all = find(ev == 2);
if isempty(toneendidx_all)
    error('No tone-end events found (event==1).');
end

% --- define tp trials anchored on tone end ---
npresamp = round(cfg.tpre * fs);
npostsamp = round(cfg.tpost * fs);

good = (toneendidx_all - npresamp >= 1) & (toneendidx_all + npostsamp <= height(traw));
toneendidx = toneendidx_all(good);
if isempty(toneendidx)
    error('After edge trimming, no complete trials remain.');
end

toneendtimes = t_sec(toneendidx);
pufftimes = t_sec(puffidx_all);

ntrialsall = numel(toneendidx);
fprintf('Found %d complete trials anchored on tone end.\n', ntrialsall);

istp = false(ntrialsall, 1);
for i = 1:ntrialsall
    t0 = toneendtimes(i);
    istp(i) = any(pufftimes > t0 & pufftimes <= (t0 + cfg.puffsearchwin));
end
fprintf('Label counts (full set): tone-only=%d, tone+puff=%d\n', sum(~istp), sum(istp));

toneendidx_tp = toneendidx(istp);
toneendtimes_tp = toneendtimes(istp);
ntp = numel(toneendidx_tp);
if ntp < 5
    error('Too few TP trials (%d) to run TP-only tone-vs-baseline test.', ntp);
end
fprintf('Using ONLY tone+puff (TP) trials: nTP=%d\n', ntp);

% --- puff lag diagnostic ---
pufflag_each = nan(ntp, 1);
for i = 1:ntp
    t0 = toneendtimes_tp(i);
    cand = pufftimes(pufftimes >= (t0 - cfg.tpre) & pufftimes <= (t0 + cfg.tpost));
    if isempty(cand), continue; end
    [~, j] = min(abs(cand - t0));
    pufflag_each(i) = cand(j) - t0;
end
goodlag = pufflag_each(isfinite(pufflag_each));
puff_rel = NaN;
if ~isempty(goodlag)
    puff_rel = median(goodlag);
    fprintf('Median puff lag among TP trials (s): %.6f\n', puff_rel);
    fprintf('Puff lag p10/p90 (s): %.6f / %.6f\n', prctile(goodlag, 10), prctile(goodlag, 90));
end
fprintf('\n');

% --- hard pre-puff cutoff for feature calculation ---
if isempty(goodlag)
    error('No finite puff lags found in TP trials; cannot enforce pre-puff-only features.');
end

puff_cut = min(goodlag);

wins_tmp = round(cfg.win_ms_feat * fs / 1000);
halfwin_feat_s_tmp = (wins_tmp - 1) / (2 * fs);

tmax_raw_time = puff_cut - 1e-12;
cut_samp = (npresamp + 1) + floor(tmax_raw_time * fs);
if cut_samp < 1
    error('cut_samp<1; puff_cut too early relative to trial start.');
end

% --- trial index matrix ---
triallensamp = npresamp + npostsamp + 1;
trialidx = zeros(ntp, triallensamp);
for i = 1:ntp
    idx0 = toneendidx_tp(i);
    trialidx(i, :) = (idx0 - npresamp):(idx0 + npostsamp);
end
rowstp = (1:ntp)';

% --- resolve rhpp12 and rhpp23 columns ---
targetswanted = {'RHPP12', 'RHPP23'};
idxtarget = zeros(1, 2);
nametarget = cell(1, 2);

for tt = 1:2
    target = targetswanted{tt};
    tgtvalid = matlab.lang.makeValidName(target, 'ReplacementStyle', 'delete');

    hit = find(strcmpi(contactcols, tgtvalid), 1, 'first');
    if isempty(hit)
        hit = find(contains(lower(contactcols), lower(target)), 1, 'first');
    end
    if isempty(hit) && contains(target, '-')
        alt = regexprep(target, '[^A-Za-z0-9]', '');
        hit = find(contains(lower(contactcols), lower(alt)), 1, 'first');
    end
    if isempty(hit)
        fprintf('\nAvailable contacts (valid names):\n');
        disp(contactcols(:)');
        error('Could not find required contact "%s" in CSV columns.', target);
    end

    idxtarget(tt) = hit;
    nametarget{tt} = contactcols{tt};
end

% preload signals for both targets
sig_bytarget = cell(1, 2);
for tt = 1:2
    sig_bytarget{tt} = double(traw.(contactcolsorig{idxtarget(tt)}));
end

% --- populate output ---
trials.traw = traw;
trials.t_sec = t_sec;
trials.ev = ev;
trials.fs = fs;
trials.contactcols = string(contactcols);
trials.contactcolsorig = string(contactcolsorig);
trials.nametarget = string(nametarget);
trials.idxtarget = idxtarget;
trials.sig_bytarget = sig_bytarget;
trials.toneendidx_tp = toneendidx_tp;
trials.toneendtimes_tp = toneendtimes_tp;
trials.ntp = ntp;
trials.trialidx = trialidx;
trials.triallensamp = triallensamp;
trials.npresamp = npresamp;
trials.npostsamp = npostsamp;
trials.rowstp = rowstp;
trials.puff_rel = puff_rel;
trials.puff_cut = puff_cut;
trials.cut_samp = cut_samp;

% --- save artifact ---
if ~exist(cfg.outdir, 'dir'), mkdir(cfg.outdir); end
saved_cfg = cfg; %#ok<NASGU>
save(matfile, 'trials', 'saved_cfg', '-v7.3');
fprintf('Saved trials_%s.mat\n', cfg.mousename);

end
