# E3 — end to end

n = 64, nu uniform, gamma = 0.05, T = 60 outer steps, inner solver (sig_L, sig_m) = (1, 100), eps_t = 0.01 (t+1)^-4.5 on the projected certificate (h^2-weighted units), cap 5000. G* = min G over 1500 minimizing-movement steps at certificate 1e-10 (cap 20000). G is normalised so that G = KL(x | q), q a density on the box.

## Target: gauss

G* = 0.002715125751; W_{2,nu}^2(x_0, rho*) = 0.031421; reference run used 40350 inner iterations, cap hit in 0 steps.

| run | gap at t=10 | gap at t=30 | gap at t=60 | total inner its | steps at cap | min gap (signed) | wall (s) |
|---|---|---|---|---|---|---|---|
| accelerated, certified eps_t | 1.281e-10 | 2.802e-16 | 3.287e-16 | 2675 | 0 | 5.985e-17 | 3 |
| non-accelerated, certified eps_t | 5.544e-13 | 5.308e-16 | 1.548e-16 | 2275 | 0 | 1.110e-16 | 2 |
| accelerated, 50 inner its | 1.179e-10 | 4.810e-16 | 3.053e-16 | 3000 | 0 | 7.633e-17 | 3 |
| non-accelerated, 50 inner its | 6.103e-13 | 1.908e-16 | 2.828e-16 | 3000 | 0 | 5.941e-17 | 3 |

- empirical slope of the accelerated certified run, log gap vs log t on t in [10, 60]: -21.96
- max_t gap/bound (accelerated, certified): 0.1552
- steps with no admissible inner iterate: accel_cert 0, mm_cert 0, accel_fixed 0, mm_fixed 0

## Target: quartic

G* = -0.003168453752; W_{2,nu}^2(x_0, rho*) = 0.011474; reference run used 39250 inner iterations, cap hit in 0 steps.

| run | gap at t=10 | gap at t=30 | gap at t=60 | total inner its | steps at cap | min gap (signed) | wall (s) |
|---|---|---|---|---|---|---|---|
| accelerated, certified eps_t | 9.993e-11 | 1.648e-16 | 1.960e-16 | 2150 | 0 | 3.123e-17 | 3 |
| non-accelerated, certified eps_t | 9.635e-14 | 1.821e-16 | 1.266e-16 | 1900 | 0 | 7.069e-17 | 3 |
| accelerated, 50 inner its | 9.908e-11 | 1.943e-16 | 1.262e-16 | 3000 | 0 | 4.770e-17 | 4 |
| non-accelerated, 50 inner its | 9.762e-14 | 6.982e-17 | 1.834e-16 | 3000 | 0 | 6.982e-17 | 4 |

- empirical slope of the accelerated certified run, log gap vs log t on t in [10, 60]: nan
- max_t gap/bound (accelerated, certified): 0.1743
- steps with no admissible inner iterate: accel_cert 0, mm_cert 0, accel_fixed 0, mm_fixed 0

# Ensemble over 20 random instances

Seed = instance: Gaussian target with centre ~ U([0.35, 0.65]^2) and variances log-uniform on [0.01, 0.06] (as in E2); nu uniform, X_0 = Id, all four runs on the same instance, each with its own reference run for G*. Tables: mean [95% bootstrap CI of the mean]; inner iterations: median [95% bootstrap CI of the median], runs that never reach the gap are right-censored. Gaps are floored at 1e-13 before taking log-ratios.

Normalised gap (G(x_t) - G*)/(G(x_0) - G*):

| run | t=10 | t=30 | t=60 |
|---|---|---|---|
| accelerated, certified eps_t | 5.15e-08 [1.39e-08, 1.13e-07] | 5.4e-16 [1.53e-16, 1.14e-15] | 1.59e-16 [1.27e-16, 1.92e-16] |
| non-accelerated, certified eps_t | 9.46e-09 [1.92e-09, 2.15e-08] | 1.54e-16 [1.2e-16, 1.93e-16] | 1.43e-16 [1.17e-16, 1.72e-16] |
| accelerated, 50 inner its | 4.92e-08 [1.39e-08, 1.06e-07] | 5.15e-16 [1.62e-16, 1.06e-15] | 1.54e-16 [1.35e-16, 1.74e-16] |
| non-accelerated, 50 inner its | 1.11e-08 [2.26e-09, 2.54e-08] | 1.48e-16 [1.19e-16, 1.84e-16] | 1.46e-16 [1.16e-16, 1.8e-16] |

Paired log10(gap accelerated / gap non-accelerated) (negative = acceleration helps):

| inner accuracy | t=10 | t=30 | t=60 |
|---|---|---|---|
| certified | 1.16 [0.892, 1.49] | 0 [0, 0] | 0 [0, 0] |
| 50 inner its | 1.07 [0.784, 1.42] | 0 [0, 0] | 0 [0, 0] |

Total inner iterations until the normalised outer gap first drops below a level:

| run | to 1e-4 | reached | to 1e-8 | reached | steps at cap (all instances) |
|---|---|---|---|---|---|
| accelerated, certified eps_t | 500 [425, 600] | 20/20 | 1.45e+03 [1.18e+03, 1.92e+03] | 20/20 | 0 |
| non-accelerated, certified eps_t | 600 [575, 675] | 20/20 | 1.3e+03 [950, 1.62e+03] | 20/20 | 0 |
| accelerated, 50 inner its | 200 [150, 200] | 20/20 | 500 [400, 550] | 20/20 | 0 |
| non-accelerated, 50 inner its | 200 [200, 250] | 20/20 | 450 [350, 500] | 20/20 | 0 |

- accelerated certified run, max_t gap/bound of Theorem 1: max over instances 0.2277, instances with ratio <= 1: 20/20
