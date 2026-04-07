%% run_lda_pipeline.m -- LDA tone-vs-baseline decoding for both patients + comparison
%
% Runs the full pipeline for SD326 (60085_39_bad) and SD327 (60085_38_good),
% then performs a between-subject permutation test comparing pre-puff decoding.

clear; clc;
tic;

%% patient definitions
patients = struct( ...
    'SD326', struct('csvfile', fullfile(pwd, 'df_neural_ted_60085_39(in).csv'), ...
                    'mousename', '60085_39_bad'), ...
    'SD327', struct('csvfile', fullfile(pwd, 'df_neural_ted_60085_38(in).csv'), ...
                    'mousename', '60085_38_good'));

%% shared output directory
outdir = fullfile(pwd, sprintf('v13pt42_OUT_%s', datestr(now, 'yyyymmdd_HHMMSS')));
if ~exist(outdir, 'dir'), mkdir(outdir); end

%% start parpool
p = gcp('nocreate');
if isempty(p)
    try
        parpool;
    catch
        fprintf('Could not start parpool; continuing without parfor.\n');
    end
end

%% run pipeline for each patient
patientNames = fieldnames(patients);
all_results = struct();

for pp = 1:numel(patientNames)
    pname = patientNames{pp};
    pat = patients.(pname);

    fprintf('\n========== %s (%s) ==========\n', pname, pat.mousename);

    cfg       = build_config(pat.mousename, pat.csvfile, outdir);
    % cfg.nperm = 2;  % uncomment for fast testing
    trials    = prepare_trials(cfg);
    features  = compute_features(cfg, trials);
    results   = run_decoding(cfg, trials, features);
    inference = run_cluster_inference(cfg, results);
    export_results(cfg, trials, results, inference);

    all_results.(pname) = results;
end

%% between-subject comparison
comparison = compare_patients(all_results.SD326, all_results.SD327);

% save comparison artifact
saved_cfg = cfg;
save(fullfile(outdir, 'comparison_327_vs_326.mat'), 'comparison', 'saved_cfg', '-v7.3');

hfig_comp = comparison.plot_results();
exportgraphics(hfig_comp, fullfile(figdir, 'comparison_results.png'), 'Resolution', 300);

fprintf('Total elapsed: %.1f s\n', toc);

%% load and plot results from disk
figdir = fullfile(pwd, 'figures');
if ~exist(figdir, 'dir'), mkdir(figdir); end

[cfg_38_good, ~, ~, results_38_good, inference_38_good] = load_pipeline_results('v13pt42_OUT_20260407_100552', '60085_38_good');
hfig_good = plot_decoding_trace(cfg_38_good, results_38_good, inference_38_good);
exportgraphics(hfig_good, fullfile(figdir, 'decoding_trace_SD327_good.png'), 'Resolution', 300);

[cfg_39_bad, ~, ~, results_39_bad, inference_39_bad] = load_pipeline_results('v13pt42_OUT_20260407_100552', '60085_39_bad');
hfig_bad = plot_decoding_trace(cfg_39_bad, results_39_bad, inference_39_bad);
exportgraphics(hfig_bad, fullfile(figdir, 'decoding_trace_SD326_bad.png'), 'Resolution', 300);


