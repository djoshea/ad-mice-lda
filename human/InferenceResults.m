classdef InferenceResults < handle
    properties
        z_obs           (:,1) double    % Z-scored observed accuracy (nwin x 1)
        z_thr           (1,1) double    % Z-score threshold for cluster detection
        munull          (1,:) double    % Null distribution mean per window (1 x nwin)
        sdnull          (1,:) double    % Null distribution std per window (1 x nwin)

        % Full-epoch clusters
        cl_start        (:,1) double    % Cluster start bin indices
        cl_end          (:,1) double    % Cluster end bin indices
        cl_mass         (:,1) double    % Cluster masses (sum of z-scores)
        cl_p            (:,1) double    % Cluster p-values
        sigranges       (:,2) double    % Significant cluster time ranges [start end] in s
        sigp            (:,1) double    % Significant cluster p-values
        sigrangesstr    (1,1) string    % Formatted string of significant ranges

        % Pre-puff-only clusters
        cl_start_pre    (:,1) double    % Pre-puff cluster start bin indices
        cl_end_pre      (:,1) double    % Pre-puff cluster end bin indices
        cl_mass_pre     (:,1) double    % Pre-puff cluster masses
        cl_p_pre        (:,1) double    % Pre-puff cluster p-values
        sigranges_pre   (:,2) double    % Pre-puff significant ranges [start end] in s
        sigp_pre        (:,1) double    % Pre-puff significant cluster p-values
        sigrangesstr_pre (1,1) string   % Formatted string of pre-puff significant ranges
    end
end
