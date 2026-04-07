classdef LdaConfig < handle
    properties
        csvfile         (1,1) string    % Path to input CSV file
        mousename       (1,1) string    % Patient identifier (e.g. '60085_38_good')
        outdir          (1,1) string    % Output directory path
        contactsetname  (1,1) string    % Contact pair name (e.g. "PAIR_RHPP12__RHPP23")
        bandsstr        (1,1) string    % Feature bands description string
        win_ms_feat     (1,1) double    % Feature window duration in ms (200)
        win_ms_dec      (1,1) double    % Decoding window duration in ms (500)
        step_ms_dec     (1,1) double    % Decoding step size in ms (50)
        step_ms_psd     (1,1) double    % PSD step size in ms (100)
        analysis_t1     (1,1) double    % Analysis start time rel tone end in s (-2.0)
        analysis_t2     (1,1) double    % Analysis end time rel tone end in s (1.65)
        delta_band      (1,2) double    % Delta bandpass range in Hz [2 4]
        theta_band      (1,2) double    % Theta bandpass range in Hz [4 8]
        alpha_band      (1,2) double    % Alpha bandpass range in Hz [8 12]
        lowbeta_band    (1,2) double    % Low-beta bandpass range in Hz [12.5 20]
        envmetric       (1,1) string    % Envelope metric: "rms" or "rect"
        env_mode        (1,1) string    % Envelope filter mode: "filter" (causal)
        env_order       (1,1) double    % Butterworth filter order for envelopes (4)
        gammashrink     (1,1) double    % LDA shrinkage parameter (0.10)
        nperm           (1,1) double    % Number of permutations (default 1000; use 2 for fast testing)
        n_cv_repeats    (1,1) double    % Number of repeated CV partitions (10)
        n_splits        (1,1) double    % Number of CV folds (5)
        tpre            (1,1) double    % Seconds before tone end for trial window (6.35)
        tpost           (1,1) double    % Seconds after tone end for trial window (5.00)
        puffsearchwin   (1,1) double    % Seconds after tone end to search for puff (3.00)
        nw              (1,1) double    % Multitaper bandwidth parameter (3)
        fmax            (1,1) double    % Max PSD frequency in Hz (50)
        base_start_default (1,1) double % Baseline window start rel tone end in s (-6.0)
        base_end_default   (1,1) double % Baseline window end rel tone end in s (-0.5)
        alpha_point     (1,1) double    % Significance level (0.05)
        random_state    (1,1) double    % RNG seed for reproducibility (42)
        fixed_cv        (1,1) logical   % Use fixed CV folds across obs + perms (true)
        fs              (1,1) double    % Forced sampling rate in Hz (500.000000011823)
        fig_base        (1,1) double    % Base figure number (31001)
        ii_cfg          (1,1) double    % Config index (11)
        fignum          (1,1) double    % Figure number = fig_base + ii_cfg - 1
        save_tifs       (1,1) logical   % Save TIF figures (true)
        save_figs       (1,1) logical   % Save MATLAB .fig files (true)
    end
end
