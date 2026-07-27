# Rebuttal numbers — NeurIPS 2026 Submission 3775

Auto-generated from `results/*/summary.json` by `collate.py`.
Values are the raw experimental outputs; paste directly into rebuttal responses.

## CONTRADICTIONS / CAVEATS (read first)

1. **λ = 0 KL targets converge exponentially, not at a power law.** Geodesic
   convexity of `KL(·‖q)` (λ ≥ 0) means `q` is log-concave, which by Bobkov's
   theorem forces a positive spectral gap, hence exponential decay of `KL(ρ_t‖q)`.
   The O(1/t) / O(1/t²) rates are *worst-case upper bounds*, approached only as the
   gap → 0 (E1a), never tight at a fixed geodesically-convex target (E1b quartic /
   cubic / uniform decay exponentially). This is the reframing already adopted in the
   revised plan; it is NOT a defect of the theorems (exponential satisfies the bound).

2. **E1c potential-only `x⁴/4`: standard slope ≈ −2, not −1.** A homogeneous
   potential `|x|^p` gives flow exponent `p/(p−2)` (p=4 → −2), and acceleration
   doubles it (acc ≈ −4). Clean power laws DO appear once the entropy/diffusion is
   removed, but the exponents are −2/−4, not the naive −1/−2. Report as such.

3. **E3 σ² = 1e-4 (σ = 0.01) is numerically intractable.** The parametric prox with
   finite-difference gradients cannot fit this ultra-sharp target (params driven to
   bounds; grid- and MC-KL both huge for all methods). Report σ² = 0.012 as primary;
   note σ² = 1e-4 as a numerical-stability limitation, not a scientific result.

4. **E7 (2-D map defect) is large, not negligible.** The index-aligned composed map
   differs from the true OT map by `δ_t/W₂ ≈ 0.44` mean / `0.94` max (quick config).
   This confirms 99SE's point that composition of OT maps is not an OT map in d ≥ 2;
   the inexact-step / Theorem-8 fix is load-bearing, not cosmetic. (1-D: δ ≈ 1e-16.)

## Solver checks (theory-response, load-bearing)

- 1-D map defect `max δ_t ≈ 1e-16` across all E1 runs (confirms the d=1 corollary).
- Momentum monotonicity violations: 0 for convex targets (quartic/cubic/gauss),
  ~0.7% for uniform, **31% for the non-convex double-well** (λ < 0) — exactly the
  absolute-continuity regime 99SE flagged; projected back by isotonic regression.

## NUMBERS


### e1a
NUMBER: e1a.2.R = 2
NUMBER: e1a.2.init_mean = 0
NUMBER: e1a.2.init_std = 2.5
NUMBER: e1a.2.initial_kl = 3.85057
NUMBER: e1a.2.measured_gap = 0.948604
NUMBER: e1a.2.sup_W2sq = 0.484759
NUMBER: e1a.2.max_delta_t_1d = 1.12595e-16
NUMBER: e1a.2.frac_steps_with_monotonicity_violation = 0
NUMBER: e1a.2.wall_seconds = 0.131294
NUMBER: e1a.2.blocks_std_1e-2 = 2
NUMBER: e1a.2.blocks_acc_1e-2 = 2
NUMBER: e1a.2.blocks_ratio_1e-2 = 1
NUMBER: e1a.2.blocks_std_1e-3 = 4
NUMBER: e1a.2.blocks_acc_1e-3 = 4
NUMBER: e1a.2.blocks_ratio_1e-3 = 1
NUMBER: e1a.4.R = 4
NUMBER: e1a.4.init_mean = 0
NUMBER: e1a.4.init_std = 3.5
NUMBER: e1a.4.initial_kl = 6.06837
NUMBER: e1a.4.measured_gap = 0.5347
NUMBER: e1a.4.sup_W2sq = 0.601953
NUMBER: e1a.4.max_delta_t_1d = 1.50948e-16
NUMBER: e1a.4.frac_steps_with_monotonicity_violation = 0
NUMBER: e1a.4.wall_seconds = 0.202298
NUMBER: e1a.4.blocks_std_1e-2 = 2
NUMBER: e1a.4.blocks_acc_1e-2 = 2
NUMBER: e1a.4.blocks_ratio_1e-2 = 1
NUMBER: e1a.4.blocks_std_1e-3 = 4
NUMBER: e1a.4.blocks_acc_1e-3 = 4
NUMBER: e1a.4.blocks_ratio_1e-3 = 1
NUMBER: e1a.8.R = 8
NUMBER: e1a.8.init_mean = 0
NUMBER: e1a.8.init_std = 5.5
NUMBER: e1a.8.initial_kl = 14.7996
NUMBER: e1a.8.measured_gap = 0.123108
NUMBER: e1a.8.sup_W2sq = 1.27196
NUMBER: e1a.8.max_delta_t_1d = 1.81363e-16
NUMBER: e1a.8.frac_steps_with_monotonicity_violation = 0
NUMBER: e1a.8.wall_seconds = 0.703662
NUMBER: e1a.8.blocks_std_1e-2 = 7
NUMBER: e1a.8.blocks_acc_1e-2 = 5
NUMBER: e1a.8.blocks_ratio_1e-2 = 1.4
NUMBER: e1a.8.blocks_std_1e-3 = 27
NUMBER: e1a.8.blocks_acc_1e-3 = 13
NUMBER: e1a.8.blocks_ratio_1e-3 = 2.07692
NUMBER: e1a.16.R = 16
NUMBER: e1a.16.init_mean = 0
NUMBER: e1a.16.init_std = 9.5
NUMBER: e1a.16.initial_kl = 61.8062
NUMBER: e1a.16.measured_gap = 0.0353118
NUMBER: e1a.16.sup_W2sq = 4.42186
NUMBER: e1a.16.max_delta_t_1d = 2.00919e-16
NUMBER: e1a.16.frac_steps_with_monotonicity_violation = 0
NUMBER: e1a.16.wall_seconds = 2.2134
NUMBER: e1a.16.blocks_std_1e-2 = 43
NUMBER: e1a.16.blocks_acc_1e-2 = 17
NUMBER: e1a.16.blocks_ratio_1e-2 = 2.52941
NUMBER: e1a.16.blocks_std_1e-3 = 114
NUMBER: e1a.16.blocks_acc_1e-3 = 27
NUMBER: e1a.16.blocks_ratio_1e-3 = 4.22222
NUMBER: e1a.32.R = 32
NUMBER: e1a.32.init_mean = 0
NUMBER: e1a.32.init_std = 17.5
NUMBER: e1a.32.initial_kl = 424.682
NUMBER: e1a.32.measured_gap = 0.00961544
NUMBER: e1a.32.sup_W2sq = 18.0109
NUMBER: e1a.32.max_delta_t_1d = 2.11771e-16
NUMBER: e1a.32.frac_steps_with_monotonicity_violation = 0
NUMBER: e1a.32.wall_seconds = 8.83351
NUMBER: e1a.32.blocks_std_1e-2 = 197
NUMBER: e1a.32.blocks_acc_1e-2 = 39
NUMBER: e1a.32.blocks_ratio_1e-2 = 5.05128
NUMBER: e1a.32.blocks_std_1e-3 = 455
NUMBER: e1a.32.blocks_acc_1e-3 = 55
NUMBER: e1a.32.blocks_ratio_1e-3 = 8.27273

### e1b_quartic
NUMBER: e1b_quartic.target = quartic
NUMBER: e1b_quartic.lam_class = 0
NUMBER: e1b_quartic.initial_gap = 418.348
NUMBER: e1b_quartic.final_kl_std = 0
NUMBER: e1b_quartic.final_kl_acc = 0
NUMBER: e1b_quartic.exp_rate_std = 1.12396
NUMBER: e1b_quartic.exp_rate_acc = 1.2395
NUMBER: e1b_quartic.slope_std_transient = nan
NUMBER: e1b_quartic.slope_acc_transient = nan
NUMBER: e1b_quartic.gap_ratio_at_T = 0
NUMBER: e1b_quartic.frac_steps_with_monotonicity_violation = 0
NUMBER: e1b_quartic.max_violation_magnitude = 0
NUMBER: e1b_quartic.max_projection_magnitude = 0
NUMBER: e1b_quartic.max_delta_t_1d = 2.04332e-16
NUMBER: e1b_quartic.sup_t_W2sq_z_rho = 27.8403
NUMBER: e1b_quartic.blocks_to_1e-3_std = 9
NUMBER: e1b_quartic.blocks_to_1e-3_acc = 5
NUMBER: e1b_quartic.blocks_to_1e-6_std = 15
NUMBER: e1b_quartic.blocks_to_1e-6_acc = 10
NUMBER: e1b_quartic.blocks_to_1e-9_std = 16
NUMBER: e1b_quartic.blocks_to_1e-9_acc = 14

### e1b_cubic
NUMBER: e1b_cubic.target = cubic
NUMBER: e1b_cubic.lam_class = 0
NUMBER: e1b_cubic.initial_gap = 71.572
NUMBER: e1b_cubic.final_kl_std = 0
NUMBER: e1b_cubic.final_kl_acc = 0
NUMBER: e1b_cubic.exp_rate_std = 1.01947
NUMBER: e1b_cubic.exp_rate_acc = 1.10142
NUMBER: e1b_cubic.slope_std_transient = nan
NUMBER: e1b_cubic.slope_acc_transient = nan
NUMBER: e1b_cubic.gap_ratio_at_T = 0
NUMBER: e1b_cubic.frac_steps_with_monotonicity_violation = 0
NUMBER: e1b_cubic.max_violation_magnitude = 0
NUMBER: e1b_cubic.max_projection_magnitude = 0
NUMBER: e1b_cubic.max_delta_t_1d = 1.69631e-16
NUMBER: e1b_cubic.sup_t_W2sq_z_rho = 27.6317
NUMBER: e1b_cubic.blocks_to_1e-3_std = 9
NUMBER: e1b_cubic.blocks_to_1e-3_acc = 6
NUMBER: e1b_cubic.blocks_to_1e-6_std = 16
NUMBER: e1b_cubic.blocks_to_1e-6_acc = 15
NUMBER: e1b_cubic.blocks_to_1e-9_std = 17
NUMBER: e1b_cubic.blocks_to_1e-9_acc = 15

### e1b_uniform
NUMBER: e1b_uniform.target = uniform
NUMBER: e1b_uniform.lam_class = 0
NUMBER: e1b_uniform.initial_gap = 2.21485e+06
NUMBER: e1b_uniform.final_kl_std = 0
NUMBER: e1b_uniform.final_kl_acc = 0
NUMBER: e1b_uniform.exp_rate_std = 1.85393
NUMBER: e1b_uniform.exp_rate_acc = 2.12072
NUMBER: e1b_uniform.slope_std_transient = nan
NUMBER: e1b_uniform.slope_acc_transient = nan
NUMBER: e1b_uniform.gap_ratio_at_T = 0
NUMBER: e1b_uniform.frac_steps_with_monotonicity_violation = 0.00666667
NUMBER: e1b_uniform.max_violation_magnitude = 0.000415285
NUMBER: e1b_uniform.max_projection_magnitude = 0.00239826
NUMBER: e1b_uniform.max_delta_t_1d = 2.21195e-16
NUMBER: e1b_uniform.sup_t_W2sq_z_rho = 28.7605
NUMBER: e1b_uniform.blocks_to_1e-3_std = 5
NUMBER: e1b_uniform.blocks_to_1e-3_acc = 4
NUMBER: e1b_uniform.blocks_to_1e-6_std = 5
NUMBER: e1b_uniform.blocks_to_1e-6_acc = 4
NUMBER: e1b_uniform.blocks_to_1e-9_std = 5
NUMBER: e1b_uniform.blocks_to_1e-9_acc = 4

### e1b_gauss_lam1e-3
NUMBER: e1b_gauss_lam1e-3.target = gauss_lam1e-3
NUMBER: e1b_gauss_lam1e-3.lam_class = positive
NUMBER: e1b_gauss_lam1e-3.initial_gap = 2.05274
NUMBER: e1b_gauss_lam1e-3.final_kl_std = 0.308458
NUMBER: e1b_gauss_lam1e-3.final_kl_acc = 0.000160396
NUMBER: e1b_gauss_lam1e-3.exp_rate_std = 0.00527834
NUMBER: e1b_gauss_lam1e-3.exp_rate_acc = 0.0334488
NUMBER: e1b_gauss_lam1e-3.slope_std_transient = nan
NUMBER: e1b_gauss_lam1e-3.slope_acc_transient = -3.90529
NUMBER: e1b_gauss_lam1e-3.gap_ratio_at_T = 1923.1
NUMBER: e1b_gauss_lam1e-3.frac_steps_with_monotonicity_violation = 0
NUMBER: e1b_gauss_lam1e-3.max_violation_magnitude = 0
NUMBER: e1b_gauss_lam1e-3.max_projection_magnitude = 0
NUMBER: e1b_gauss_lam1e-3.max_delta_t_1d = 2.03081e-16
NUMBER: e1b_gauss_lam1e-3.sup_t_W2sq_z_rho = 872.858
NUMBER: e1b_gauss_lam1e-3.blocks_to_1e-3_std = -1
NUMBER: e1b_gauss_lam1e-3.blocks_to_1e-3_acc = 194
NUMBER: e1b_gauss_lam1e-3.blocks_to_1e-6_std = -1
NUMBER: e1b_gauss_lam1e-3.blocks_to_1e-6_acc = -1
NUMBER: e1b_gauss_lam1e-3.blocks_to_1e-9_std = -1
NUMBER: e1b_gauss_lam1e-3.blocks_to_1e-9_acc = -1

### e1b_doublewell_np
NUMBER: e1b_doublewell_np.target = doublewell_np
NUMBER: e1b_doublewell_np.lam_class = negative
NUMBER: e1b_doublewell_np.initial_gap = 14.3025
NUMBER: e1b_doublewell_np.final_kl_std = 4.45194e-07
NUMBER: e1b_doublewell_np.final_kl_acc = 0
NUMBER: e1b_doublewell_np.exp_rate_std = 0.0436331
NUMBER: e1b_doublewell_np.exp_rate_acc = 0.140465
NUMBER: e1b_doublewell_np.slope_std_transient = -4.43057
NUMBER: e1b_doublewell_np.slope_acc_transient = -8.16307
NUMBER: e1b_doublewell_np.gap_ratio_at_T = 4.45194e+293
NUMBER: e1b_doublewell_np.frac_steps_with_monotonicity_violation = 0.313333
NUMBER: e1b_doublewell_np.max_violation_magnitude = 0.135508
NUMBER: e1b_doublewell_np.max_projection_magnitude = 3.71264
NUMBER: e1b_doublewell_np.max_delta_t_1d = 2.00706e-16
NUMBER: e1b_doublewell_np.sup_t_W2sq_z_rho = 25.7291
NUMBER: e1b_doublewell_np.blocks_to_1e-3_std = 142
NUMBER: e1b_doublewell_np.blocks_to_1e-3_acc = 59
NUMBER: e1b_doublewell_np.blocks_to_1e-6_std = 293
NUMBER: e1b_doublewell_np.blocks_to_1e-6_acc = 81
NUMBER: e1b_doublewell_np.blocks_to_1e-9_std = -1
NUMBER: e1b_doublewell_np.blocks_to_1e-9_acc = 81

### e1c_potential
NUMBER: e1c_potential.functional = potential_only
NUMBER: e1c_potential.slope_std_potential_only = -2.16449
NUMBER: e1c_potential.slope_acc_potential_only = -3.75857
NUMBER: e1c_potential.final_std = 2.8631e-06
NUMBER: e1c_potential.final_acc = 4.86189e-10
NUMBER: e1c_potential.frac_steps_with_monotonicity_violation = 0.893333
NUMBER: e1c_potential.max_violation_magnitude = 0.00805524
NUMBER: e1c_potential.max_projection_magnitude = 0.0504734
NUMBER: e1c_potential.max_delta_t_1d = 2.51157e-16
NUMBER: e1c_potential.sup_t_W2sq_z_rho = 31.2338

### e1c_interaction
NUMBER: e1c_interaction.functional = interaction
NUMBER: e1c_interaction.slope_std_interaction = -2.18309
NUMBER: e1c_interaction.slope_acc_interaction = -3.51446
NUMBER: e1c_interaction.final_std = 1.77769e-07
NUMBER: e1c_interaction.final_acc = 5.36316e-10
NUMBER: e1c_interaction.frac_steps_with_monotonicity_violation = 0.33
NUMBER: e1c_interaction.max_violation_magnitude = 0.00383113
NUMBER: e1c_interaction.max_projection_magnitude = 0.0116153
NUMBER: e1c_interaction.max_delta_t_1d = 1.77442e-16
NUMBER: e1c_interaction.sup_t_W2sq_z_rho = 31.2338

### e2_fig2_gaussian
NUMBER: e2_fig2_gaussian.crossover_iter_lambda0.04 = 869
NUMBER: e2_fig2_gaussian.crossover_iter_lambda1 = 24
NUMBER: e2_fig2_gaussian.final_kl_std_lam0.04 = 6.31089e-30
NUMBER: e2_fig2_gaussian.final_kl_acc_lam0.04 = 2.46519e-32
NUMBER: e2_fig2_gaussian.init = N(5.0,2.5^2)

### e2_fig4_doublewell
NUMBER: e2_fig4_doublewell.final_kl_std = 1.62278e-10
NUMBER: e2_fig4_doublewell.final_kl_acc = 1.1477e-11
NUMBER: e2_fig4_doublewell.min_kl_acc = 1.1477e-11
NUMBER: e2_fig4_doublewell.init = N(0.5,2.0^2)

### e3_mixture_sig2_0p012
NUMBER: e3_mixture_sig2_0p012.final_KL_std_init1 = 0.0953098
NUMBER: e3_mixture_sig2_0p012.iters_to_mode_separation_std_init1 = -1
NUMBER: e3_mixture_sig2_0p012.final_KL_acc_init1 = 0.0953104
NUMBER: e3_mixture_sig2_0p012.iters_to_mode_separation_acc_init1 = -1
NUMBER: e3_mixture_sig2_0p012.final_KL_gd_init1 = 0.806369
NUMBER: e3_mixture_sig2_0p012.iters_to_mode_separation_gd_init1 = -1
NUMBER: e3_mixture_sig2_0p012.gd_best_eta_init1 = 0.03
NUMBER: e3_mixture_sig2_0p012.max_traj_divergence_init1 = 0.00164623
NUMBER: e3_mixture_sig2_0p012.final_KL_std_init2 = 0.0953112
NUMBER: e3_mixture_sig2_0p012.iters_to_mode_separation_std_init2 = -1
NUMBER: e3_mixture_sig2_0p012.final_KL_acc_init2 = 0.0953112
NUMBER: e3_mixture_sig2_0p012.iters_to_mode_separation_acc_init2 = -1
NUMBER: e3_mixture_sig2_0p012.final_KL_gd_init2 = 0.805716
NUMBER: e3_mixture_sig2_0p012.iters_to_mode_separation_gd_init2 = -1
NUMBER: e3_mixture_sig2_0p012.gd_best_eta_init2 = 0.03
NUMBER: e3_mixture_sig2_0p012.max_traj_divergence_init2 = 0.000440393
NUMBER: e3_mixture_sig2_0p012.final_KL_std_init3 = 0.0953103
NUMBER: e3_mixture_sig2_0p012.iters_to_mode_separation_std_init3 = -1
NUMBER: e3_mixture_sig2_0p012.final_KL_acc_init3 = 0.0953102
NUMBER: e3_mixture_sig2_0p012.iters_to_mode_separation_acc_init3 = -1
NUMBER: e3_mixture_sig2_0p012.final_KL_gd_init3 = 0.0957634
NUMBER: e3_mixture_sig2_0p012.iters_to_mode_separation_gd_init3 = -1
NUMBER: e3_mixture_sig2_0p012.gd_best_eta_init3 = 0.01
NUMBER: e3_mixture_sig2_0p012.max_traj_divergence_init3 = 0.00102994

### e4_walltime
NUMBER: e4_walltime.sec_per_block_std_e1a = 0.00777525
NUMBER: e4_walltime.sec_per_block_acc_e1a = 0.00953297
NUMBER: e4_walltime.momentum_overhead_pct_e1a = 0.108651
NUMBER: e4_walltime.sec_per_block_std_e1b_cubic = 0.00337538
NUMBER: e4_walltime.sec_per_block_acc_e1b_cubic = 0.0052382
NUMBER: e4_walltime.momentum_overhead_pct_e1b_cubic = 0.263201
NUMBER: e4_walltime.sec_per_block_std_e1b_doublewell_np = 0.0817612
NUMBER: e4_walltime.sec_per_block_acc_e1b_doublewell_np = 0.0595873
NUMBER: e4_walltime.momentum_overhead_pct_e1b_doublewell_np = 0.666693
NUMBER: e4_walltime.sec_per_block_std_e1b_gauss_lam1e-3 = 0.0611097
NUMBER: e4_walltime.sec_per_block_acc_e1b_gauss_lam1e-3 = 0.0515198
NUMBER: e4_walltime.momentum_overhead_pct_e1b_gauss_lam1e-3 = 0.0328695
NUMBER: e4_walltime.sec_per_block_std_e1b_quartic = 0.00404902
NUMBER: e4_walltime.sec_per_block_acc_e1b_quartic = 0.00706615
NUMBER: e4_walltime.momentum_overhead_pct_e1b_quartic = 0.194089
NUMBER: e4_walltime.sec_per_block_std_e1b_uniform = 0.00449837
NUMBER: e4_walltime.sec_per_block_acc_e1b_uniform = 0.00772623
NUMBER: e4_walltime.momentum_overhead_pct_e1b_uniform = 0.256219
NUMBER: e4_walltime.sec_per_block_std_e1c_interaction = 0.261988
NUMBER: e4_walltime.sec_per_block_acc_e1c_interaction = 0.0838036
NUMBER: e4_walltime.momentum_overhead_pct_e1c_interaction = 0.0603455
NUMBER: e4_walltime.sec_per_block_std_e1c_potential = 0.0244238
NUMBER: e4_walltime.sec_per_block_acc_e1c_potential = 0.00514441
NUMBER: e4_walltime.momentum_overhead_pct_e1c_potential = 2.59987

### e56_quartic
NUMBER: e56_quartic.target = quartic
NUMBER: e56_quartic.final_gap_acc = 0
NUMBER: e56_quartic.restart_count_function = 1
NUMBER: e56_quartic.final_gap_restart_function = 0
NUMBER: e56_quartic.restart_count_gradient = 23
NUMBER: e56_quartic.final_gap_restart_gradient = 0
NUMBER: e56_quartic.better_restart_rule = function
NUMBER: e56_quartic.restart_count = 1
NUMBER: e56_quartic.final_gap_restart = 0
NUMBER: e56_quartic.fallback_frac = 0.00333333
NUMBER: e56_quartic.final_gap_safeguard = 0

### e56_gauss_lam1e-3
NUMBER: e56_gauss_lam1e-3.target = gauss_lam1e-3
NUMBER: e56_gauss_lam1e-3.final_gap_acc = 0.000160396
NUMBER: e56_gauss_lam1e-3.restart_count_function = 1
NUMBER: e56_gauss_lam1e-3.final_gap_restart_function = 2.67807e-05
NUMBER: e56_gauss_lam1e-3.restart_count_gradient = 1
NUMBER: e56_gauss_lam1e-3.final_gap_restart_gradient = 2.56252e-05
NUMBER: e56_gauss_lam1e-3.better_restart_rule = gradient
NUMBER: e56_gauss_lam1e-3.restart_count = 1
NUMBER: e56_gauss_lam1e-3.final_gap_restart = 2.56252e-05
NUMBER: e56_gauss_lam1e-3.fallback_frac = 0.00333333
NUMBER: e56_gauss_lam1e-3.final_gap_safeguard = 2.7546e-05

### e7_defect2d
NUMBER: e7_defect2d.delta_t_normalized_max = 1.79264
NUMBER: e7_defect2d.delta_t_normalized_mean = 1.16124
NUMBER: e7_defect2d.accumulated_defect_term = 16.0011
NUMBER: e7_defect2d.blocks = 25
NUMBER: e7_defect2d.particles = 12000
NUMBER: e7_defect2d.n_sub = 2048

### e8_1_exactprox
NUMBER: e8_1_exactprox.slope_asga_explicit = -2.01658
NUMBER: e8_1_exactprox.slope_asga_exactprox = -1.63568
NUMBER: e8_1_exactprox.inner_iters_per_outer_mean = 14.825
NUMBER: e8_1_exactprox.final_gap_explicit = 1.44568e-08
NUMBER: e8_1_exactprox.final_gap_exactprox = 1.36041e-09

### e8_2_lipschitz
NUMBER: e8_2_lipschitz.L_empirical = 0.0152475
NUMBER: e8_2_lipschitz.L_certified = 72.9813
NUMBER: e8_2_lipschitz.L_ratio = 4786.46

### e8_3_sweep
NUMBER: e8_3_sweep.slope_sga_eps0.002_n60_m2 = -0.954803
NUMBER: e8_3_sweep.slope_asga_eps0.002_n60_m2 = -3.41783
NUMBER: e8_3_sweep.slope_sga_eps0.002_n60_m4 = -0.987085
NUMBER: e8_3_sweep.slope_asga_eps0.002_n60_m4 = -2.42484
NUMBER: e8_3_sweep.slope_sga_eps0.002_n60_m8 = -0.985427
NUMBER: e8_3_sweep.slope_asga_eps0.002_n60_m8 = -2.11405
NUMBER: e8_3_sweep.slope_sga_eps0.002_n120_m2 = -0.960866
NUMBER: e8_3_sweep.slope_asga_eps0.002_n120_m2 = -3.4237
NUMBER: e8_3_sweep.slope_sga_eps0.002_n120_m4 = -0.994552
NUMBER: e8_3_sweep.slope_asga_eps0.002_n120_m4 = -2.38323
NUMBER: e8_3_sweep.slope_sga_eps0.002_n120_m8 = -0.988345
NUMBER: e8_3_sweep.slope_asga_eps0.002_n120_m8 = -2.09823
NUMBER: e8_3_sweep.slope_sga_eps0.002_n240_m2 = -0.967825
NUMBER: e8_3_sweep.slope_asga_eps0.002_n240_m2 = -3.50135
NUMBER: e8_3_sweep.slope_sga_eps0.002_n240_m4 = -0.999674
NUMBER: e8_3_sweep.slope_asga_eps0.002_n240_m4 = -2.33496
NUMBER: e8_3_sweep.slope_sga_eps0.002_n240_m8 = -0.987517
NUMBER: e8_3_sweep.slope_asga_eps0.002_n240_m8 = -2.12776
NUMBER: e8_3_sweep.slope_sga_eps0.004_n60_m2 = -0.956324
NUMBER: e8_3_sweep.slope_asga_eps0.004_n60_m2 = -1.56937
NUMBER: e8_3_sweep.slope_sga_eps0.004_n60_m4 = -0.994195
NUMBER: e8_3_sweep.slope_asga_eps0.004_n60_m4 = -1.83852
NUMBER: e8_3_sweep.slope_sga_eps0.004_n60_m8 = -0.992845
NUMBER: e8_3_sweep.slope_asga_eps0.004_n60_m8 = -1.88361
NUMBER: e8_3_sweep.slope_sga_eps0.004_n120_m2 = -0.962547
NUMBER: e8_3_sweep.slope_asga_eps0.004_n120_m2 = -1.58537
NUMBER: e8_3_sweep.slope_sga_eps0.004_n120_m4 = -0.997188
NUMBER: e8_3_sweep.slope_asga_eps0.004_n120_m4 = -1.84374
NUMBER: e8_3_sweep.slope_sga_eps0.004_n120_m8 = -0.994626
NUMBER: e8_3_sweep.slope_asga_eps0.004_n120_m8 = -1.88898
NUMBER: e8_3_sweep.slope_sga_eps0.004_n240_m2 = -0.96269
NUMBER: e8_3_sweep.slope_asga_eps0.004_n240_m2 = -1.59449
NUMBER: e8_3_sweep.slope_sga_eps0.004_n240_m4 = -0.997348
NUMBER: e8_3_sweep.slope_asga_eps0.004_n240_m4 = -1.84603
NUMBER: e8_3_sweep.slope_sga_eps0.004_n240_m8 = -0.994815
NUMBER: e8_3_sweep.slope_asga_eps0.004_n240_m8 = -1.89029
NUMBER: e8_3_sweep.slope_sga_eps0.008_n60_m2 = -0.967441
NUMBER: e8_3_sweep.slope_asga_eps0.008_n60_m2 = -1.93238
NUMBER: e8_3_sweep.slope_sga_eps0.008_n60_m4 = -1.00901
NUMBER: e8_3_sweep.slope_asga_eps0.008_n60_m4 = -1.99979
NUMBER: e8_3_sweep.slope_sga_eps0.008_n60_m8 = -1.01594
NUMBER: e8_3_sweep.slope_asga_eps0.008_n60_m8 = -2.02176
NUMBER: e8_3_sweep.slope_sga_eps0.008_n120_m2 = -0.970105
NUMBER: e8_3_sweep.slope_asga_eps0.008_n120_m2 = -1.93081
NUMBER: e8_3_sweep.slope_sga_eps0.008_n120_m4 = -1.01048
NUMBER: e8_3_sweep.slope_asga_eps0.008_n120_m4 = -1.99892
NUMBER: e8_3_sweep.slope_sga_eps0.008_n120_m8 = -1.01706
NUMBER: e8_3_sweep.slope_asga_eps0.008_n120_m8 = -2.02109
NUMBER: e8_3_sweep.slope_sga_eps0.008_n240_m2 = -0.969633
NUMBER: e8_3_sweep.slope_asga_eps0.008_n240_m2 = -1.93315
NUMBER: e8_3_sweep.slope_sga_eps0.008_n240_m4 = -1.01021
NUMBER: e8_3_sweep.slope_asga_eps0.008_n240_m4 = -2.00024
NUMBER: e8_3_sweep.slope_sga_eps0.008_n240_m8 = -1.01714
NUMBER: e8_3_sweep.slope_asga_eps0.008_n240_m8 = -2.02202
NUMBER: e8_3_sweep.slope_sga_eps0.016_n60_m2 = -1.03695
NUMBER: e8_3_sweep.slope_asga_eps0.016_n60_m2 = -2.03613
NUMBER: e8_3_sweep.slope_sga_eps0.016_n60_m4 = -1.05609
NUMBER: e8_3_sweep.slope_asga_eps0.016_n60_m4 = -2.07055
NUMBER: e8_3_sweep.slope_sga_eps0.016_n60_m8 = -1.06178
NUMBER: e8_3_sweep.slope_asga_eps0.016_n60_m8 = -2.09
NUMBER: e8_3_sweep.slope_sga_eps0.016_n120_m2 = -1.03712
NUMBER: e8_3_sweep.slope_asga_eps0.016_n120_m2 = -2.03863
NUMBER: e8_3_sweep.slope_sga_eps0.016_n120_m4 = -1.0562
NUMBER: e8_3_sweep.slope_asga_eps0.016_n120_m4 = -2.07083
NUMBER: e8_3_sweep.slope_sga_eps0.016_n120_m8 = -1.06228
NUMBER: e8_3_sweep.slope_asga_eps0.016_n120_m8 = -2.08831
NUMBER: e8_3_sweep.slope_sga_eps0.016_n240_m2 = -1.03637
NUMBER: e8_3_sweep.slope_asga_eps0.016_n240_m2 = -2.04223
NUMBER: e8_3_sweep.slope_sga_eps0.016_n240_m4 = -1.05568
NUMBER: e8_3_sweep.slope_asga_eps0.016_n240_m4 = -2.07337
NUMBER: e8_3_sweep.slope_sga_eps0.016_n240_m8 = -1.06224
NUMBER: e8_3_sweep.slope_asga_eps0.016_n240_m8 = -2.08979

### e9_iso_quartic
NUMBER: e9_iso_quartic.blocks_to_thresh_std_d2 = 7
NUMBER: e9_iso_quartic.blocks_to_thresh_acc_d2 = 6
NUMBER: e9_iso_quartic.walltime_to_thresh_std_d2 = 20.1513
NUMBER: e9_iso_quartic.walltime_to_thresh_acc_d2 = 16.8432
NUMBER: e9_iso_quartic.final_W2_std_d2 = 0.037005
NUMBER: e9_iso_quartic.final_W2_acc_d2 = 0.0361328
NUMBER: e9_iso_quartic.blocks_to_thresh_std_d5 = -1
NUMBER: e9_iso_quartic.blocks_to_thresh_acc_d5 = -1
NUMBER: e9_iso_quartic.walltime_to_thresh_std_d5 = -1
NUMBER: e9_iso_quartic.walltime_to_thresh_acc_d5 = -1
NUMBER: e9_iso_quartic.final_W2_std_d5 = 0.275302
NUMBER: e9_iso_quartic.final_W2_acc_d5 = 0.276208
NUMBER: e9_iso_quartic.blocks_to_thresh_std_d10 = -1
NUMBER: e9_iso_quartic.blocks_to_thresh_acc_d10 = -1
NUMBER: e9_iso_quartic.walltime_to_thresh_std_d10 = -1
NUMBER: e9_iso_quartic.walltime_to_thresh_acc_d10 = -1
NUMBER: e9_iso_quartic.final_W2_std_d10 = 0.661493
NUMBER: e9_iso_quartic.final_W2_acc_d10 = 0.66192
NUMBER: e9_iso_quartic.blocks_to_thresh_std_d20 = -1
NUMBER: e9_iso_quartic.blocks_to_thresh_acc_d20 = -1
NUMBER: e9_iso_quartic.walltime_to_thresh_std_d20 = -1
NUMBER: e9_iso_quartic.walltime_to_thresh_acc_d20 = -1
NUMBER: e9_iso_quartic.final_W2_std_d20 = 1.16794
NUMBER: e9_iso_quartic.final_W2_acc_d20 = 1.16775
NUMBER: e9_iso_quartic.blocks_to_thresh_std_d50 = -1
NUMBER: e9_iso_quartic.blocks_to_thresh_acc_d50 = -1
NUMBER: e9_iso_quartic.walltime_to_thresh_std_d50 = -1
NUMBER: e9_iso_quartic.walltime_to_thresh_acc_d50 = -1
NUMBER: e9_iso_quartic.final_W2_std_d50 = 1.92377
NUMBER: e9_iso_quartic.final_W2_acc_d50 = 1.92373

### e9_flatvalley
NUMBER: e9_flatvalley.blocks_to_thresh_std_d2 = 22
NUMBER: e9_flatvalley.blocks_to_thresh_acc_d2 = 11
NUMBER: e9_flatvalley.walltime_to_thresh_std_d2 = 61.0517
NUMBER: e9_flatvalley.walltime_to_thresh_acc_d2 = 30.6505
NUMBER: e9_flatvalley.final_W2_std_d2 = 0.0912614
NUMBER: e9_flatvalley.final_W2_acc_d2 = 0.0771212
NUMBER: e9_flatvalley.blocks_to_thresh_std_d5 = -1
NUMBER: e9_flatvalley.blocks_to_thresh_acc_d5 = -1
NUMBER: e9_flatvalley.walltime_to_thresh_std_d5 = -1
NUMBER: e9_flatvalley.walltime_to_thresh_acc_d5 = -1
NUMBER: e9_flatvalley.final_W2_std_d5 = 0.542409
NUMBER: e9_flatvalley.final_W2_acc_d5 = 0.541895
NUMBER: e9_flatvalley.blocks_to_thresh_std_d10 = -1
NUMBER: e9_flatvalley.blocks_to_thresh_acc_d10 = -1
NUMBER: e9_flatvalley.walltime_to_thresh_std_d10 = -1
NUMBER: e9_flatvalley.walltime_to_thresh_acc_d10 = -1
NUMBER: e9_flatvalley.final_W2_std_d10 = 1.24511
NUMBER: e9_flatvalley.final_W2_acc_d10 = 1.24471
NUMBER: e9_flatvalley.blocks_to_thresh_std_d20 = -1
NUMBER: e9_flatvalley.blocks_to_thresh_acc_d20 = -1
NUMBER: e9_flatvalley.walltime_to_thresh_std_d20 = -1
NUMBER: e9_flatvalley.walltime_to_thresh_acc_d20 = -1
NUMBER: e9_flatvalley.final_W2_std_d20 = 2.05587
NUMBER: e9_flatvalley.final_W2_acc_d20 = 2.05514
NUMBER: e9_flatvalley.blocks_to_thresh_std_d50 = -1
NUMBER: e9_flatvalley.blocks_to_thresh_acc_d50 = -1
NUMBER: e9_flatvalley.walltime_to_thresh_std_d50 = -1
NUMBER: e9_flatvalley.walltime_to_thresh_acc_d50 = -1
NUMBER: e9_flatvalley.final_W2_std_d50 = 3.08542
NUMBER: e9_flatvalley.final_W2_acc_d50 = 3.0855

### e11_outer_ring
NUMBER: e11_outer_ring.final_w2_std = 0.586824
NUMBER: e11_outer_ring.final_w2_acc = 0.427892
NUMBER: e11_outer_ring.reduction_pct = 27.0834

### e3_mixture
NUMBER: e3_mixture.primary_sig2_0.012.final_KL_std_init1 = 0.0953098
NUMBER: e3_mixture.primary_sig2_0.012.iters_to_mode_separation_std_init1 = -1
NUMBER: e3_mixture.primary_sig2_0.012.final_KL_acc_init1 = 0.0953104
NUMBER: e3_mixture.primary_sig2_0.012.iters_to_mode_separation_acc_init1 = -1
NUMBER: e3_mixture.primary_sig2_0.012.final_KL_gd_init1 = 0.806369
NUMBER: e3_mixture.primary_sig2_0.012.iters_to_mode_separation_gd_init1 = -1
NUMBER: e3_mixture.primary_sig2_0.012.gd_best_eta_init1 = 0.03
NUMBER: e3_mixture.primary_sig2_0.012.max_traj_divergence_init1 = 0.00164623
NUMBER: e3_mixture.primary_sig2_0.012.final_KL_std_init2 = 0.0953112
NUMBER: e3_mixture.primary_sig2_0.012.iters_to_mode_separation_std_init2 = -1
NUMBER: e3_mixture.primary_sig2_0.012.final_KL_acc_init2 = 0.0953112
NUMBER: e3_mixture.primary_sig2_0.012.iters_to_mode_separation_acc_init2 = -1
NUMBER: e3_mixture.primary_sig2_0.012.final_KL_gd_init2 = 0.805716
NUMBER: e3_mixture.primary_sig2_0.012.iters_to_mode_separation_gd_init2 = -1
NUMBER: e3_mixture.primary_sig2_0.012.gd_best_eta_init2 = 0.03
NUMBER: e3_mixture.primary_sig2_0.012.max_traj_divergence_init2 = 0.000440393
NUMBER: e3_mixture.primary_sig2_0.012.final_KL_std_init3 = 0.0953103
NUMBER: e3_mixture.primary_sig2_0.012.iters_to_mode_separation_std_init3 = -1
NUMBER: e3_mixture.primary_sig2_0.012.final_KL_acc_init3 = 0.0953102
NUMBER: e3_mixture.primary_sig2_0.012.iters_to_mode_separation_acc_init3 = -1
NUMBER: e3_mixture.primary_sig2_0.012.final_KL_gd_init3 = 0.0957634
NUMBER: e3_mixture.primary_sig2_0.012.iters_to_mode_separation_gd_init3 = -1
NUMBER: e3_mixture.primary_sig2_0.012.gd_best_eta_init3 = 0.01
NUMBER: e3_mixture.primary_sig2_0.012.max_traj_divergence_init3 = 0.00102994
