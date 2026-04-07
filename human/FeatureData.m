classdef FeatureData < handle
    properties
        featmat         (:,:,:) double  % Feature matrix (ntp x ntimebins x nfeat)
        featnames       (1,:) string    % Feature names (e.g. "RHPP12_alphaPSD")
        featorder       (1,:) cell      % Ordered feature type names
        cs_all          (:,:,:) double  % Cumulative sums for fast window means (ntp x ntimebins+1 x nfeat)
        splitsrep       (:,1) cell      % Repeated CV partition structs (n_cv_repeats x 1)
        ktrial          (1,1) double    % Number of CV folds actually used
        times_trial     (1,:) double    % Feature bin center times rel tone end in s
        basemask        (1,:) logical   % Logical mask for baseline time bins
        starts          (1,:) double    % Feature window start sample indices
        ntimebins       (1,1) double    % Number of feature time bins
        dt_bin          (1,1) double    % Feature bin spacing in s
        t0_bin          (1,1) double    % Time of first feature bin center in s
        nbins           (1,1) double    % Total number of feature bins (same as ntimebins)
        wins            (1,1) double    % Feature window length in samples
        nfeat           (1,1) double    % Total number of features (nc * nfeatperc)
        nfeatperc       (1,1) double    % Number of features per contact
        nc              (1,1) double    % Number of contacts used
    end
end
