function cfg = build_config(mousename, csvfile, outdir)
% BUILD_CONFIG  Create an LdaConfig with production default parameters.
%
%   cfg = build_config(mousename, csvfile, outdir)
%
% Callers can override fields after construction, e.g. cfg.nperm = 2;

cfg = LdaConfig();

cfg.csvfile    = string(csvfile);
cfg.mousename  = string(mousename);
cfg.outdir     = string(outdir);

% single config (fig 31011)
cfg.contactsetname = "PAIR_RHPP12__RHPP23";
cfg.bandsstr       = "alphaPSD + lowbetaPSD + deltaENV + thetaENV";

cfg.win_ms_feat = 200;
cfg.win_ms_dec  = 500;

cfg.analysis_t1 = -2.000;
cfg.analysis_t2 =  1.650;

cfg.delta_band = [2 4];
cfg.envmetric  = "rms";
cfg.env_order  = 4;
cfg.gammashrink = 0.100;

% decoding step
cfg.step_ms_dec = 50;

% permutations
cfg.nperm = 1000;

% repeated cv partitions
cfg.n_cv_repeats = 10;

% trial definition
cfg.tpre          = 6.35;
cfg.tpost         = 5.00;
cfg.puffsearchwin = 3.00;

% psd settings
cfg.step_ms_psd = 100;
cfg.nw          = 3;
cfg.fmax        = 50;
cfg.alpha_band    = [8 12];
cfg.lowbeta_band  = [12.5 20];

% envelope settings
cfg.env_mode   = "filter";
cfg.theta_band = [4 8];

% baseline pseudo-anchor region rel tone end
cfg.base_start_default = -6.0;
cfg.base_end_default   = -0.5;

% cv / permutation
cfg.n_splits     = 5;
cfg.alpha_point  = 0.05;
cfg.random_state = 42;
cfg.fixed_cv     = true;

% figure settings
cfg.fig_base  = 31001;
cfg.save_tifs = true;
cfg.save_figs = true;

% fig 31011 is config index 11 when fig_base=31001
cfg.ii_cfg = 11;
cfg.fignum = cfg.fig_base + (cfg.ii_cfg - 1);

% forced sampling rate
cfg.fs = 500.000000011823;

end
