classdef ComparisonResults < handle
    properties
        delta_obs       (1,1) double    % Observed difference: mean_327(good) - mean_326(bad)
        delta_null      (:,1) double    % Null distribution of differences (nperm x 1)
        p_twosided      (1,1) double    % Two-sided p-value (H1: |good| ~= |bad|
        obs_mean_326    (1,1) double    % SD326 (bad) mean post-tone pre-puff accuracy
        obs_mean_327    (1,1) double    % SD327 (good) mean post-tone pre-puff accuracy
        nperm           (1,1) double    % Number of permutations used
    end

    methods
        function hfig = plot_results(obj)
            % PLOT_RESULTS  Histogram of null delta with observed value marked.
            hfig = figure; clf;
            set(hfig, 'Color', 'w');
            hold on;

            histogram(obj.delta_null, 'Normalization', 'count', ...
                'FaceColor', [0.7 0.7 0.7], 'EdgeColor', 'w');
            xline(obj.delta_obs, 'r-', 'LineWidth', 2);

            xlabel('\Delta bACC (good - bad)');
            ylabel('Count');
            title(sprintf('Between-subject permutation test (p = %.4f, n_{perm} = %d)', ...
                obj.p_twosided, obj.nperm), 'Interpreter', 'tex');
            legend({'Null distribution', sprintf('Observed \\Delta = %.4f', obj.delta_obs)}, ...
                'Location', 'best');
            grid on; box on;
            drawnow;
        end
    end
end
