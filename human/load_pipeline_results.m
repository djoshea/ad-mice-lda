function [cfg, trials, features, results, inference] = load_pipeline_results(outdir, mousename)
% LOAD_PIPELINE_RESULTS  Load cached .mat artifacts for a given patient.
%
%   [cfg, trials, features, results, inference] = load_pipeline_results(outdir, mousename)
%
% Loads whichever artifacts exist in outdir. Missing stages return empty [].
%
% Example:
%   [cfg, ~, ~, results, inference] = load_pipeline_results('v13pt42_OUT_20260407/', '60085_38_good');
%   plot_decoding_trace(cfg, results, inference);

cfg       = [];
trials    = [];
features  = [];
results   = [];
inference = [];

names = {'trials', 'features', 'decoding', 'inference'};

for i = 1:numel(names)
    matfile = fullfile(outdir, sprintf('%s_%s.mat', names{i}, mousename));
    if ~exist(matfile, 'file')
        fprintf('Not found: %s\n', matfile);
        continue;
    end

    tmp = load(matfile);

    % recover cfg from whichever artifact we find first
    if isempty(cfg) && isfield(tmp, 'saved_cfg')
        cfg = tmp.saved_cfg;
    end

    switch names{i}
        case 'trials'
            trials = tmp.trials;
        case 'features'
            features = tmp.features;
        case 'decoding'
            results = tmp.results;
        case 'inference'
            inference = tmp.inference;
    end

    fprintf('Loaded %s\n', matfile);
end

if isempty(cfg)
    warning('No cached artifacts found for mouse "%s" in %s', mousename, outdir);
end

end
