classdef TonePuffTrialData < handle
    properties
        traw            table           % Raw CSV data table
        t_sec           (:,1) double    % Uniform timebase in seconds (n_samples x 1)
        ev              (:,1) double    % Event codes (1=tone end, 2=puff)
        fs              (1,1) double    % Sampling rate in Hz
        contactcols     (1,:) string    % Valid MATLAB names for contact columns
        contactcolsorig (1,:) string    % Original contact column names from CSV
        nametarget      (1,:) string    % Target contact names (valid MATLAB names)
        idxtarget       (1,:) double    % Column indices of target contacts in CSV
        sig_bytarget    (1,:) cell      % Cell array of continuous signals per target contact
        toneendidx_tp   (:,1) double    % Sample indices of tone-end events for TP trials
        toneendtimes_tp (:,1) double    % Times of tone-end events for TP trials in s
        ntp             (1,1) double    % Number of tone+puff (TP) trials
        trialidx        (:,:) double    % Trial-to-sample index matrix (ntp x triallensamp)
        triallensamp    (1,1) double    % Number of samples per trial
        npresamp        (1,1) double    % Number of pre-tone-end samples per trial
        npostsamp       (1,1) double    % Number of post-tone-end samples per trial
        rowstp          (:,1) double    % Row indices 1:ntp (convenience)
        puff_rel        (1,1) double    % Median puff lag rel tone end in s
        puff_cut        (1,1) double    % Earliest puff lag (hard safety cutoff) in s
        cut_samp        (1,1) double    % Last allowed sample index in trial segment
    end
end
