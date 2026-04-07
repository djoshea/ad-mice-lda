function hfig = plot_decoding_trace(cfg, results, inference)
% PLOT_DECODING_TRACE  Plot balanced accuracy vs time with CI, null, and cluster shading.
%
%   hfig = plot_decoding_trace(cfg, results, inference)
%
% Inputs:
%   cfg       - LdaConfig
%   results   - DecodingResults (needs bacc_obs, bacc_ci_lo, bacc_ci_hi, null_bacc, times_end)
%   inference - InferenceResults (needs cl_start, cl_end, cl_p, cl_p_pre)
%
% Returns:
%   hfig - figure handle

times_end = results.times_end;
bacc_obs  = results.bacc_obs;
ci_lo     = results.bacc_ci_lo;
ci_hi     = results.bacc_ci_hi;
null_bacc = results.null_bacc;
puff_rel  = results.puff_rel;

cl_start = inference.cl_start;
cl_end   = inference.cl_end;
cl_p     = inference.cl_p;
cl_p_pre = inference.cl_p_pre;

null_mean = mean(null_bacc, 1)';
null_p95  = prctile(null_bacc, 95, 1)';

% --- figure ---
hfig = figure; clf;
set(hfig, 'Color', 'w', 'Name', sprintf('%s — %s', cfg.mousename, cfg.contactsetname));
hold on;

% CI band
x = times_end(:);
x2 = [x; flipud(x)];
y2 = [ci_lo(:); flipud(ci_hi(:))];
patch(x2, y2, 'k', 'FaceAlpha', 0.10, 'EdgeColor', 'none');

% cluster shading
yl = [0 1];
if ~isempty(cl_start)
    for kk = 1:numel(cl_start)
        x0 = times_end(max(1, cl_start(kk)));
        x1 = times_end(min(numel(times_end), cl_end(kk)));
        if cl_p(kk) < 0.05, fa = 0.15; else, fa = 0.05; end
        patch([x0 x1 x1 x0], [yl(1) yl(1) yl(2) yl(2)], 'k', 'FaceAlpha', fa, 'EdgeColor', 'none');
    end
end

% traces
plot(times_end, bacc_obs, 'LineWidth', 2);
plot(times_end, null_mean, '--', 'LineWidth', 1.5);
plot(times_end, null_p95, ':', 'LineWidth', 1.5);

% reference lines
xline(0, '-.', 'LineWidth', 1.4);
if isfinite(puff_rel), xline(puff_rel, '-.', 'LineWidth', 1.4); end
win_dec_s = cfg.win_ms_dec / 1000;
xline(cfg.analysis_t1 + win_dec_s, ':', 'LineWidth', 1.2);
xline(cfg.analysis_t2, ':', 'LineWidth', 1.2);

xlabel('Time relative to alignment (s) [decoding window END]');
ylabel('Balanced accuracy');

bestclp = NaN;
if ~isempty(cl_p), bestclp = min(cl_p); end
bestclp_pre = NaN;
if ~isempty(cl_p_pre), bestclp_pre = min(cl_p_pre); end

ttl = sprintf('%s | %s | %s | feat=%d dec=%d | bestP=%.5g | bestP_{pre}=%.5g', ...
    cfg.mousename, cfg.contactsetname, cfg.bandsstr, ...
    cfg.win_ms_feat, cfg.win_ms_dec, bestclp, bestclp_pre);
title(ttl, 'Interpreter', 'none');

ylim([0 1]);
xlim([min(times_end)-0.2, max(times_end)+0.2]);
grid on; box on;
legend({'95% CI (CV repeats)', 'Observed bACC (fixed folds)', 'Null mean', 'Null 95th'}, ...
    'Location', 'southwest');
drawnow;

end
