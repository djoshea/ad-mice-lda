function comparison = compare_patients(results_326, results_327)
% COMPARE_PATIENTS  Between-subject permutation test on pre-puff decoding accuracy.
%
%   comparison = compare_patients(results_326, results_327)
%
% Tests whether SD327 (good) has higher pre-puff decoding accuracy than SD326 (bad).
% Uses the null traces from each patient's permutation test, paired by index.
%
% Inputs:
%   results_326 - DecodingResults for SD326 (60085_39_bad)
%   results_327 - DecodingResults for SD327 (60085_38_good)
%
% Returns a ComparisonResults object.

comparison = ComparisonResults();

% validate matching permutation counts
nperm_326 = size(results_326.null_bacc, 1);
nperm_327 = size(results_327.null_bacc, 1);
assert(nperm_326 == nperm_327, ...
    'Permutation counts must match: SD326 has %d, SD327 has %d', nperm_326, nperm_327);
nperm = nperm_326;

% post-tone-end AND pre-puff mask per patient (times_end >= 0 & mpre)
mask_326 = results_326.mpre & (results_326.times_end >= 0);
mask_327 = results_327.mpre & (results_327.times_end >= 0);

% observed mean post-tone pre-puff accuracy per patient
obs_mean_326 = mean(results_326.bacc_obs(mask_326));
obs_mean_327 = mean(results_327.bacc_obs(mask_327));
% delta = good - bad (positive means SD327 has higher accuracy)
delta_obs = obs_mean_327 - obs_mean_326;

% null distribution: pair permutations by index
delta_null = zeros(nperm, 1);
for i = 1:nperm
    null_mean_326_i = mean(results_326.null_bacc(i, mask_326'));
    null_mean_327_i = mean(results_327.null_bacc(i, mask_327'));
    delta_null(i) = null_mean_327_i - null_mean_326_i;
end

% H0 = good ~ bad
% H1 = |good| != |bad| (continuity correction matches cluster-mass convention)
% two-sided p-value: test |delta| in either direction
p_twosided = (sum(abs(delta_null) >= abs(delta_obs)) + 1) / (nperm + 1);

% populate output
comparison.delta_obs    = delta_obs;
comparison.delta_null   = delta_null;
comparison.p_twosided   = p_twosided;
comparison.obs_mean_326 = obs_mean_326;
comparison.obs_mean_327 = obs_mean_327;
comparison.nperm        = nperm;

% report
fprintf('\n=== Between-subject comparison (post-tone-end, pre-puff epoch) ===\n');
fprintf('SD326 (bad)  time bins: %d (of %d pre-puff)\n', sum(mask_326), sum(results_326.mpre));
fprintf('SD327 (good) time bins: %d (of %d pre-puff)\n', sum(mask_327), sum(results_327.mpre));
fprintf('SD326 (bad)  mean bACC: %.4f\n', obs_mean_326);
fprintf('SD327 (good) mean bACC: %.4f\n', obs_mean_327);
fprintf('Delta (good - bad):      %.4f\n', delta_obs);
fprintf('Null delta mean +/- std: %.4f +/- %.4f\n', mean(delta_null), std(delta_null));
fprintf('p-value (two-sided):     %.6f\n', p_twosided);
fprintf('nperm:                   %d\n', nperm);
fprintf('===================================================\n\n');

end
