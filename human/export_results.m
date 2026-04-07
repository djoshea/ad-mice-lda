function summary = export_results(cfg, trials, results, inference)
% EXPORT_RESULTS  Alt CI bands, publication figure, Prism CSV, cluster CSVs, summary stats.
%
%   summary = export_results(cfg, trials, results, inference)

% --- unpack ---
ntp          = trials.ntp;
puff_rel     = trials.puff_rel;
nametarget   = trials.nametarget;
bacc_obs     = results.bacc_obs;
null_bacc    = results.null_bacc;
times_end    = results.times_end;
times_center = results.times_center;
xobs         = results.xobs;
y0           = results.y0;
nwin         = results.nwin;
gammashrink  = cfg.gammashrink;
random_state = cfg.random_state;

% config tag for file naming
cfgtagbase = sprintf('cfg%02d_%s_feat%d_dec%d_A1%.2f_A2%.2f_delta%.1f_%.1f_env%s_ord%d_g%.2f', ...
    cfg.ii_cfg, char(cfg.contactsetname), cfg.win_ms_feat, cfg.win_ms_dec, ...
    cfg.analysis_t1, cfg.analysis_t2, cfg.delta_band(1), cfg.delta_band(2), ...
    char(cfg.envmetric), cfg.env_order, cfg.gammashrink);
cfgtagbase = regexprep(cfgtagbase, '[^\w\-\+\=\.\s]', '');
cfgtagbase = strrep(cfgtagbase, ' ', '_');
cfgtagbase = regexprep(cfgtagbase, '_+', '_');

outdir_cfg = fullfile(cfg.outdir, cfgtagbase);
if ~exist(outdir_cfg, 'dir'), mkdir(outdir_cfg); end

% --- repeated CV for 95% CI bands (fixed anchors) ---
ktrial = min(cfg.n_splits, ntp);
n_cv_repeats = cfg.n_cv_repeats;
bacc_rep = nan(n_cv_repeats, nwin);

for rr = 1:n_cv_repeats
    rng(random_state + 10000 + rr);
    cvr = cvpartition(ntp, 'KFold', ktrial);
    splitsr = cell(cvr.NumTestSets, 1);

    for kk = 1:cvr.NumTestSets
        tetrials = find(test(cvr, kk));
        trtrials = find(training(cvr, kk));
        teidx = [tetrials; tetrials + ntp];
        tridx = [trtrials; trtrials + ntp];
        splitsr{kk} = struct('train', tridx, 'test', teidx);
    end

    bacc_rr = nan(nwin, 1);
    for wi = 1:nwin
        xw = squeeze(xobs(:, wi, :));
        yhat_all = nan(size(y0));

        for kk = 1:numel(splitsr)
            tridx = splitsr{kk}.train;
            teidx = splitsr{kk}.test;

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
        bacc_rr(wi) = 0.5 * (tpr + tnr);
    end

    bacc_rep(rr, :) = bacc_rr(:)';

    if mod(rr, max(1, round(n_cv_repeats/5))) == 0 || rr == 1 || rr == n_cv_repeats
        fprintf('  CV repeat %d/%d\n', rr, n_cv_repeats);
    end
end

ci_lo = prctile(bacc_rep, 2.5, 1)';
ci_hi = prctile(bacc_rep, 97.5, 1)';

% --- unpack inference ---
z_obs         = inference.z_obs;
cl_start      = inference.cl_start;
cl_end        = inference.cl_end;
cl_mass       = inference.cl_mass;
cl_p          = inference.cl_p;
cl_start_pre  = inference.cl_start_pre;
cl_end_pre    = inference.cl_end_pre;
cl_p_pre      = inference.cl_p_pre;
sigranges     = inference.sigranges;
sigp          = inference.sigp;
sigrangesstr  = inference.sigrangesstr;
sigranges_pre = inference.sigranges_pre;
sigp_pre      = inference.sigp_pre;
sigrangesstr_pre = inference.sigrangesstr_pre;

% --- figure ---
hfig = plot_decoding_trace(cfg, results, inference);

bestclp = NaN;
if ~isempty(cl_p), bestclp = min(cl_p); end
bestclp_pre = NaN;
if ~isempty(cl_p_pre), bestclp_pre = min(cl_p_pre); end

% save figs
fignum = cfg.fignum;
tiffile = ""; figfile = "";
if cfg.save_tifs
    safe = sprintf('FIG%05d_%s_%s', fignum, cfg.mousename, cfgtagbase);
    safe = regexprep(safe, '[^\w\-\+\=\.\s]', '');
    safe = strrep(safe, ' ', '_');
    safe = regexprep(safe, '_+', '_');
    if numel(safe) > 180, safe = safe(1:180); end
    tiffile = string(fullfile(outdir_cfg, sprintf('%s.tif', safe)));
    try
        exportgraphics(hfig, char(tiffile), 'Resolution', 300);
    catch
        try, print(hfig, char(tiffile), '-dtiff', '-r300'); catch, end
    end
end

if cfg.save_figs
    figfile = string(fullfile(outdir_cfg, sprintf('FIG%05d_%s.fig', fignum, cfgtagbase)));
    try
        savefig(hfig, char(figfile));
    catch
    end
end

% --- prism CSV ---
sigmaskfull = zeros(nwin, 1);
if ~isempty(cl_start)
    for kk = 1:numel(cl_start)
        if cl_p(kk) < 0.05
            s0 = max(1, cl_start(kk));
            s1 = min(nwin, cl_end(kk));
            sigmaskfull(s0:s1) = 1;
        end
    end
end

sigmaskpre = zeros(nwin, 1);
if ~isempty(cl_start_pre)
    for kk = 1:numel(cl_start_pre)
        if cl_p_pre(kk) < 0.05
            s0 = max(1, cl_start_pre(kk));
            s1 = min(nwin, cl_end_pre(kk));
            sigmaskpre(s0:s1) = 1;
        end
    end
end

prismtable = table( ...
    times_center(:), times_end(:), ...
    bacc_obs(:), ci_lo(:), ci_hi(:), ...
    null_mean(:), null_p95(:), ...
    z_obs(:), sigmaskfull, sigmaskpre, ...
    'VariableNames', { ...
    'time_center_s', 'time_end_s', ...
    'bacc_obs', 'bacc_ci_low', 'bacc_ci_high', ...
    'null_mean', 'null_p95', ...
    'z_obs', 'sig_full', 'sig_pre'});

prismcsv = string(fullfile(outdir_cfg, sprintf('FIG%05d_%s_PrismData.csv', fignum, cfgtagbase)));
writetable(prismtable, char(prismcsv));

% --- cluster CSVs ---
% full sig clusters
if isempty(sigranges)
    sigtable = table(zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
        'VariableNames', {'cluster_index', 't_start_end', 't_end_end', 'duration_s', 'cluster_p'});
else
    cluster_index = (1:size(sigranges,1))';
    t_start_end = sigranges(:,1);
    t_end_end   = sigranges(:,2);
    duration_s  = t_end_end - t_start_end;
    cluster_p   = sigp;
    sigtable = table(cluster_index, t_start_end, t_end_end, duration_s, cluster_p, ...
        'VariableNames', {'cluster_index', 't_start_end', 't_end_end', 'duration_s', 'cluster_p'});
end

% all clusters
if isempty(cl_start)
    clustertable = table(zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
        'VariableNames', {'cluster_index', 'start_bin', 'end_bin', 'mass', 'p'});
else
    cluster_index = (1:numel(cl_start))';
    clustertable = table(cluster_index, cl_start(:), cl_end(:), cl_mass(:), cl_p(:), ...
        'VariableNames', {'cluster_index', 'start_bin', 'end_bin', 'mass', 'p'});
end

sigcsv     = string(fullfile(outdir_cfg, sprintf('FIG%05d_%s_SigClusters.csv', fignum, cfgtagbase)));
allclustcsv = string(fullfile(outdir_cfg, sprintf('FIG%05d_%s_AllClusters.csv', fignum, cfgtagbase)));

writetable(sigtable, char(sigcsv));
writetable(clustertable, char(allclustcsv));

% pre-puff sig clusters
if isempty(sigranges_pre)
    sigtablepre = table(zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), zeros(0,1), ...
        'VariableNames', {'cluster_index', 't_start_end', 't_end_end', 'duration_s', 'cluster_p'});
else
    cluster_index = (1:size(sigranges_pre,1))';
    t_start_end = sigranges_pre(:,1);
    t_end_end   = sigranges_pre(:,2);
    duration_s  = t_end_end - t_start_end;
    cluster_p   = sigp_pre;
    sigtablepre = table(cluster_index, t_start_end, t_end_end, duration_s, cluster_p, ...
        'VariableNames', {'cluster_index', 't_start_end', 't_end_end', 'duration_s', 'cluster_p'});
end

sigprecsv = string(fullfile(outdir_cfg, sprintf('FIG%05d_%s_SigClusters_PRE.csv', fignum, cfgtagbase)));
writetable(sigtablepre, char(sigprecsv));

% --- summary stats row ---
usetargets = [1 2];
contactlist = strjoin(string(nametarget(usetargets)), "+");
nsig = size(sigranges, 1);
nsigpre = size(sigranges_pre, 1);

statsrow = table( ...
    cfg.ii_cfg, fignum, string(cfg.csvfile), string(cfg.mousename), ...
    string(cfg.contactsetname), string(contactlist), string(cfg.bandsstr), ...
    cfg.fs, ntp, puff_rel, ...
    cfg.win_ms_feat, cfg.win_ms_dec, cfg.step_ms_dec, ...
    cfg.analysis_t1, cfg.analysis_t2, ...
    cfg.base_start_default, cfg.base_end_default, ...
    cfg.delta_band(1), cfg.delta_band(2), cfg.theta_band(1), cfg.theta_band(2), ...
    cfg.envmetric, cfg.env_order, cfg.gammashrink, ...
    cfg.nperm, cfg.n_splits, cfg.fixed_cv, cfg.n_cv_repeats, ...
    bestclp, bestclp_pre, nsig, nsigpre, ...
    sigrangesstr_pre, sigrangesstr, ...
    string(tiffile), string(figfile), string(prismcsv), ...
    string(sigcsv), string(sigprecsv), string(allclustcsv), ...
    'VariableNames', { ...
    'cfgIndex', 'figNum', 'csvFile', 'mousename', ...
    'contactSet', 'contactsUsed', 'bandsUsed', ...
    'Fs', 'nTP', 'puff_rel_median', ...
    'featWin_ms', 'decWin_ms', 'stepDec_ms', ...
    'analysisT1', 'analysisT2', ...
    'baseStart', 'baseEnd', ...
    'delta_lo', 'delta_hi', 'theta_lo', 'theta_hi', ...
    'envMetric', 'ENV_ORDER', 'gammaShrink', ...
    'nperm', 'N_SPLITS', 'fixed_cv', 'N_CV_REPEATS', ...
    'bestClusterP', 'bestClusterP_pre', 'nSig', 'nSig_pre', ...
    'sigRangesStr_pre', 'sigRangesStr', ...
    'tifFile', 'figFile', 'prismCsv', ...
    'sigCsv', 'sigPreCsv', 'allClustersCsv'});

fprintf('DONE FIG %d | bestP=%.5g | bestP_pre=%.5g | nSig=%d | nSig_pre=%d | preRanges=%s\n', ...
    fignum, bestclp, bestclp_pre, nsig, nsigpre, sigrangesstr_pre);

% --- output struct ---
summary = statsrow;

% --- save artifact ---
matfile = fullfile(cfg.outdir, sprintf('summary_%s.mat', cfg.mousename));
saved_cfg = cfg; %#ok<NASGU>
save(matfile, 'summary', 'saved_cfg', '-v7.3');
fprintf('Saved summary_%s.mat\n', cfg.mousename);

end
