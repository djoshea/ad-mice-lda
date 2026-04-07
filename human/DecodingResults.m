classdef DecodingResults < handle
    properties
        bacc_obs        (:,1) double    % Observed balanced accuracy trace (nwin x 1)
        bacc_ci_lo      (:,1) double    % 2.5th percentile CI lower bound (nwin x 1)
        bacc_ci_hi      (:,1) double    % 97.5th percentile CI upper bound (nwin x 1)
        null_bacc       (:,:) double    % Null balanced accuracy traces (nperm x nwin)
        times_end       (:,1) double    % Decoding window end times in s (nwin x 1)
        times_center    (:,1) double    % Decoding window center times in s (nwin x 1)
        winbounds       (:,2) double    % Decoding window [start end] times (nwin x 2)
        nwin            (1,1) double    % Number of decoding time windows
        mpre            (:,1) logical   % Pre-puff epoch mask (nwin x 1)
        puff_rel        (1,1) double    % Median puff lag rel tone end in s
        anchors_obs     (:,1) double    % Random baseline anchor offsets per trial (ntp x 1)
        xobs            (:,:,:) double  % Observed feature matrix (2*ntp x nwin x nfeat)
        x_tone          (:,:,:) double  % Tone-locked features (ntp x nwin x nfeat)
        y0              (:,1) double    % Binary labels: 0=baseline, 1=tone (2*ntp x 1)
    end
end
