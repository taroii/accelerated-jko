# E3 — end to end

n = 64, nu uniform, gamma = 0.0005, T = 300 outer steps, inner solver (sig_L, sig_m) = (1, 100), eps_t = 0.01 (t+1)^-4.5 on the projected certificate (h^2-weighted units), cap 5000; a certified solve also ends ('stalled') once its coefficients move by less than 1e-10 (relative) over 25 iterations. G* = min G over 1500 minimizing-movement steps with gamma = 0.05 at certificate 1e-10 (cap 20000). G is normalised so that G = KL(x | q), q a density on the box.

## Target: gauss

G* = 0.002715125751; W_{2,nu}^2(x_0, rho*) = 0.031421; reference run used 40350 inner iterations, cap hit in 0 steps.

| run | gap at t=20 | gap at t=50 | gap at t=100 | gap at t=300 | total inner its | steps at cap | steps stalled | min gap (signed) | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| accelerated, certified eps_t | 6.330e-03 | 3.937e-04 | 2.819e-07 | 8.809e-10 | 39925 | 0 | 273 | 4.525e-12 | 51 |
| non-accelerated, certified eps_t | 2.776e-01 | 2.768e-02 | 1.080e-03 | 1.495e-08 | 40825 | 0 | 281 | 1.495e-08 | 52 |
| accelerated, 50 inner its | 6.296e-03 | 3.862e-04 | 2.798e-07 | 8.689e-10 | 15000 | 0 | 0 | 4.232e-12 | 19 |
| non-accelerated, 50 inner its | 2.776e-01 | 2.769e-02 | 1.080e-03 | 1.496e-08 | 15000 | 0 | 0 | 1.496e-08 | 19 |

- empirical slope of the accelerated certified run, log gap vs log t on t in [10, 300]: -6.40
- max_t gap/bound (accelerated, certified): 0.2878
- steps with no admissible inner iterate: accel_cert 0, mm_cert 0, accel_fixed 0, mm_fixed 0

## Target: quartic

G* = -0.003168453752; W_{2,nu}^2(x_0, rho*) = 0.011474; reference run used 39250 inner iterations, cap hit in 0 steps.

| run | gap at t=20 | gap at t=50 | gap at t=100 | gap at t=300 | total inner its | steps at cap | steps stalled | min gap (signed) | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| accelerated, certified eps_t | 2.577e-04 | 1.767e-04 | 5.330e-06 | 9.296e-11 | 34575 | 0 | 286 | 3.908e-13 | 57 |
| non-accelerated, certified eps_t | 9.009e-02 | 1.334e-02 | 5.522e-04 | 1.522e-09 | 36175 | 0 | 284 | 1.522e-09 | 60 |
| accelerated, 50 inner its | 2.577e-04 | 1.766e-04 | 5.324e-06 | 9.276e-11 | 15000 | 0 | 0 | 3.724e-13 | 24 |
| non-accelerated, 50 inner its | 9.009e-02 | 1.334e-02 | 5.522e-04 | 1.522e-09 | 15000 | 0 | 0 | 1.522e-09 | 25 |

- empirical slope of the accelerated certified run, log gap vs log t on t in [10, 300]: -6.26
- max_t gap/bound (accelerated, certified): 0.2396
- steps with no admissible inner iterate: accel_cert 0, mm_cert 0, accel_fixed 0, mm_fixed 0

# Ensemble over 20 random instances

Seed = instance: Gaussian target with centre ~ U([0.35, 0.65]^2) and variances log-uniform on [0.01, 0.06] (as in E2); nu uniform, X_0 = Id, all four runs on the same instance, each with its own reference run for G*. Tables: mean [95% bootstrap CI of the mean]; inner iterations: median [95% bootstrap CI of the median], runs that never reach the gap are right-censored. Gaps are floored at 1e-13 before taking log-ratios.

Normalised gap (G(x_t) - G*)/(G(x_0) - G*):

| run | t=20 | t=50 | t=100 | t=300 |
|---|---|---|---|---|
| accelerated, certified eps_t | 0.0166 [0.0104, 0.0251] | 0.000597 [0.000308, 0.00101] | 1.95e-05 [1.14e-05, 2.87e-05] | 3.32e-08 [1.26e-08, 5.9e-08] |
| non-accelerated, certified eps_t | 0.24 [0.203, 0.278] | 0.0433 [0.0284, 0.0608] | 0.0067 [0.00339, 0.0112] | 4.35e-05 [1.23e-05, 9.42e-05] |
| accelerated, 50 inner its | 0.0166 [0.0104, 0.0251] | 0.000594 [0.000306, 0.00101] | 1.93e-05 [1.12e-05, 2.84e-05] | 3.28e-08 [1.24e-08, 5.85e-08] |
| non-accelerated, 50 inner its | 0.24 [0.203, 0.278] | 0.0433 [0.0284, 0.0608] | 0.00671 [0.00339, 0.0112] | 4.35e-05 [1.23e-05, 9.43e-05] |

Paired log10(gap accelerated / gap non-accelerated) (negative = acceleration helps):

| inner accuracy | t=20 | t=50 | t=100 | t=300 |
|---|---|---|---|---|
| certified | -1.28 [-1.38, -1.17] | -1.99 [-2.1, -1.89] | -2.44 [-2.62, -2.28] | -2.6 [-3.04, -2.15] |
| 50 inner its | -1.28 [-1.38, -1.17] | -2 [-2.11, -1.9] | -2.45 [-2.63, -2.29] | -2.61 [-3.05, -2.16] |

Total inner iterations until the normalised outer gap first drops below a level:

| run | to 0.01 | reached | to 0.0001 | reached | to 1e-06 | reached | steps at cap (all instances) |
|---|---|---|---|---|---|---|---|
| accelerated, certified eps_t | 3.18e+03 [2.72e+03, 3.95e+03] | 20/20 | 9.9e+03 [8.18e+03, 1.08e+04] | 20/20 | 1.97e+04 [1.84e+04, 2.26e+04] | 20/20 | 0 |
| non-accelerated, certified eps_t | 1.24e+04 [1.05e+04, 1.49e+04] | 20/20 | 3.48e+04 [2.38e+04, 3.98e+04] | 19/20 | inf [3.88e+04, inf] | 9/20 | 0 |
| accelerated, 50 inner its | 1.05e+03 [900, 1.15e+03] | 20/20 | 2.8e+03 [2.35e+03, 2.95e+03] | 20/20 | 5.75e+03 [5.4e+03, 6.65e+03] | 20/20 | 0 |
| non-accelerated, 50 inner its | 3.75e+03 [3.05e+03, 4.3e+03] | 20/20 | 1.06e+04 [7.35e+03, 1.27e+04] | 19/20 | inf [1.27e+04, inf] | 9/20 | 0 |

- accelerated certified run, max_t gap/bound of Theorem 1: max over instances 0.2924, instances with ratio <= 1: 20/20
