function inference = run_cluster_inference(cfg, results)
% RUN_CLUSTER_INFERENCE  Z-score, cluster-mass tests (full + pre-puff), format sig ranges.
%
%   inference = run_cluster_inference(cfg, results)
%
% Checks for cached artifact at <outdir>/inference_<mousename>.mat.

inference = InferenceResults();

% --- cache check ---
matfile = fullfile(cfg.outdir, sprintf('inference_%s.mat', cfg.mousename));
if exist(matfile, 'file')
    tmp = load(matfile);
    if isfield(tmp, 'saved_cfg') && isequal(tmp.saved_cfg, cfg)
        flds = fieldnames(tmp.inference);
        for fi = 1:numel(flds)
            inference.(flds{fi}) = tmp.inference.(flds{fi});
        end
        fprintf('Loaded cached inference_%s.mat\n', cfg.mousename);
        return;
    end
end

% --- unpack ---
bacc_obs  = results.bacc_obs;
null_bacc = results.null_bacc;
times_end = results.times_end;
mpre      = results.mpre;
nperm     = cfg.nperm;

% --- z-score observed vs null ---
munull = mean(null_bacc, 1);
sdnull = std(null_bacc, 0, 1); sdnull(sdnull < 1e-12) = 1e-12;

z_obs = (bacc_obs(:)' - munull) ./ sdnull;
z_thr = norminv(1 - cfg.alpha_point);

above = (z_obs > z_thr);

% --- cluster detection (full) ---
cl_start = []; cl_end = []; cl_mass = [];
jj = 1;
while jj <= numel(z_obs)
    if above(jj)
        kk2 = jj; mass = 0;
        while kk2 <= numel(z_obs) && above(kk2)
            mass = mass + z_obs(kk2);
            kk2 = kk2 + 1;
        end
        cl_start(end+1) = jj; %#ok<AGROW>
        cl_end(end+1)   = kk2 - 1; %#ok<AGROW>
        cl_mass(end+1)  = mass; %#ok<AGROW>
        jj = kk2;
    else
        jj = jj + 1;
    end
end

maxmassnull = zeros(nperm, 1);
for pi = 1:nperm
    zpi = (null_bacc(pi, :) - munull) ./ sdnull;
    abovepi = (zpi > z_thr);

    jj = 1; best = 0;
    while jj <= numel(zpi)
        if abovepi(jj)
            kk2 = jj; mass = 0;
            while kk2 <= numel(zpi) && abovepi(kk2)
                mass = mass + zpi(kk2);
                kk2 = kk2 + 1;
            end
            best = max(best, mass);
            jj = kk2;
        else
            jj = jj + 1;
        end
    end
    maxmassnull(pi) = best;
end

cl_p = nan(numel(cl_mass), 1);
for kk = 1:numel(cl_mass)
    cl_p(kk) = (sum(maxmassnull >= cl_mass(kk)) + 1) / (nperm + 1);
end

% significant ranges (full)
sigranges = []; sigp = [];
if ~isempty(cl_start)
    for kk = 1:numel(cl_start)
        if cl_p(kk) < 0.05
            s0 = max(1, cl_start(kk));
            s1 = min(numel(times_end), cl_end(kk));
            sigranges = [sigranges; times_end(s0) times_end(s1)]; %#ok<AGROW>
            sigp = [sigp; cl_p(kk)]; %#ok<AGROW>
        end
    end
end

% --- cluster detection (pre-puff only) ---
above_pre = above & mpre';

cl_start_pre = []; cl_end_pre = []; cl_mass_pre = [];
jj = 1;
while jj <= numel(z_obs)
    if above_pre(jj)
        kk2 = jj; mass = 0;
        while kk2 <= numel(z_obs) && above_pre(kk2)
            mass = mass + z_obs(kk2);
            kk2 = kk2 + 1;
        end
        cl_start_pre(end+1) = jj; %#ok<AGROW>
        cl_end_pre(end+1)   = kk2 - 1; %#ok<AGROW>
        cl_mass_pre(end+1)  = mass; %#ok<AGROW>
        jj = kk2;
    else
        jj = jj + 1;
    end
end

maxmassnull_pre = zeros(nperm, 1);
for pi = 1:nperm
    zpi = (null_bacc(pi, :) - munull) ./ sdnull;
    abovepi = (zpi > z_thr);
    abovepi_pre = abovepi & mpre';

    jj = 1; best = 0;
    while jj <= numel(zpi)
        if abovepi_pre(jj)
            kk2 = jj; mass = 0;
            while kk2 <= numel(zpi) && abovepi_pre(kk2)
                mass = mass + zpi(kk2);
                kk2 = kk2 + 1;
            end
            best = max(best, mass);
            jj = kk2;
        else
            jj = jj + 1;
        end
    end
    maxmassnull_pre(pi) = best;
end

cl_p_pre = nan(numel(cl_mass_pre), 1);
for kk = 1:numel(cl_mass_pre)
    cl_p_pre(kk) = (sum(maxmassnull_pre >= cl_mass_pre(kk)) + 1) / (nperm + 1);
end

% significant ranges (pre)
sigranges_pre = []; sigp_pre = [];
if ~isempty(cl_start_pre)
    for kk = 1:numel(cl_start_pre)
        if cl_p_pre(kk) < 0.05
            s0 = max(1, cl_start_pre(kk));
            s1 = min(numel(times_end), cl_end_pre(kk));
            sigranges_pre = [sigranges_pre; times_end(s0) times_end(s1)]; %#ok<AGROW>
            sigp_pre = [sigp_pre; cl_p_pre(kk)]; %#ok<AGROW>
        end
    end
end

% --- format range strings ---
if isempty(sigranges)
    sigrangesstr = "none";
else
    parts = strings(size(sigranges, 1), 1);
    for rr = 1:size(sigranges, 1)
        parts(rr) = sprintf('[%.3f %.3f](p=%.6g)', sigranges(rr,1), sigranges(rr,2), sigp(rr));
    end
    sigrangesstr = strjoin(parts, '; ');
end

if isempty(sigranges_pre)
    sigrangesstr_pre = "none";
else
    parts = strings(size(sigranges_pre, 1), 1);
    for rr = 1:size(sigranges_pre, 1)
        parts(rr) = sprintf('[%.3f %.3f](p=%.6g)', sigranges_pre(rr,1), sigranges_pre(rr,2), sigp_pre(rr));
    end
    sigrangesstr_pre = strjoin(parts, '; ');
end

% --- populate output ---
inference.z_obs    = z_obs(:);
inference.z_thr    = z_thr;
inference.munull   = munull;
inference.sdnull   = sdnull;

inference.cl_start = cl_start(:);
inference.cl_end   = cl_end(:);
inference.cl_mass  = cl_mass(:);
inference.cl_p     = cl_p(:);
inference.sigranges    = sigranges;
inference.sigp         = sigp;
inference.sigrangesstr = sigrangesstr;

inference.cl_start_pre = cl_start_pre(:);
inference.cl_end_pre   = cl_end_pre(:);
inference.cl_mass_pre  = cl_mass_pre(:);
inference.cl_p_pre     = cl_p_pre(:);
inference.sigranges_pre    = sigranges_pre;
inference.sigp_pre         = sigp_pre;
inference.sigrangesstr_pre = sigrangesstr_pre;

% --- save artifact ---
if ~exist(cfg.outdir, 'dir'), mkdir(cfg.outdir); end
saved_cfg = cfg; %#ok<NASGU>
save(matfile, 'inference', 'saved_cfg', '-v7.3');
fprintf('Saved inference_%s.mat\n', cfg.mousename);

end
