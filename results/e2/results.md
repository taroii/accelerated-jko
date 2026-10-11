# E2 — inner solver

n = 64, gamma = 0.05, theta = 0.99, N = 3000 iterations per run, reference J* from 20000 preconditioned iterations (ratio 10). Gaps and certificates are in h^2-weighted units (J h^2).

## Target: gauss

Iterations to gap < threshold (None = not reached in 3000):
| run | sig_L | sig_m | <0.01 | <0.0001 | <1e-06 | <1e-08 | final gap | time (s) |
|---|---|---|---|---|---|---|---|---|
| scalar | 1.27e-05 | 1.27e-05 | None | None | None | None | 1.024e+00 | 3.3 |
| precond r=0.1 | 1 | 0.1 | 735 | 1706 | 2979 | None | 9.299e-07 | 3.4 |
| precond r=1 | 1 | 1 | 80 | 170 | 297 | 440 | 8.882e-16 | 3.2 |
| precond r=10 | 1 | 10 | 19 | 48 | 74 | 102 | 8.882e-16 | 3.0 |
| precond r=100 | 1 | 100 | 14 | 46 | 78 | 112 | 4.441e-16 | 3.0 |
| precond (0.1, 1) | 0.1 | 1 | 62 | 171 | 303 | 447 | 4.441e-16 | 3.2 |
| precond (10, 100) | 10 | 100 | 117 | 433 | 748 | 1076 | 1.332e-15 | 3.4 |
| balanced eq.(15) | 1.374 | 91.44 | 18 | 62 | 106 | 152 | 8.882e-16 | 3.1 |
| n=128, r=100 | 1 | 100 | 19 | 113 | 199 | 286 | 8.882e-16 | 9.8 |

Certificate / true gap at the first certificate evaluation (every 25 its) after each threshold is crossed — projected certificate (spec certificate in parentheses):
| run | <0.01 | <0.0001 | <1e-06 | <1e-08 |
|---|---|---|---|---|
| scalar | None | None | None | None |
| precond r=0.1 | 8.37e+00 (2.94e+02) | 3.10e+01 (3.50e+04) | 1.92e+02 (3.52e+06) | None |
| precond r=1 | 6.53e+01 (9.12e+02) | 2.83e+02 (4.03e+04) | 1.68e+03 (3.71e+06) | 1.79e+04 (4.49e+08) |
| precond r=10 | 1.85e+02 (8.19e+02) | 2.06e+03 (4.98e+04) | 1.77e+04 (3.98e+06) | 9.92e+05 (1.71e+10) |
| precond r=100 | 2.82e+02 (1.65e+03) | 1.85e+03 (6.46e+04) | 5.37e+04 (6.68e+07) | 2.83e+05 (1.94e+09) |
| precond (0.1, 1) | 1.82e+01 (4.37e+02) | 4.45e+01 (3.71e+04) | 2.93e+02 (6.88e+06) | 1.75e+03 (3.69e+08) |
| precond (10, 100) | 1.07e+02 (3.95e+02) | 1.49e+03 (4.23e+04) | 1.24e+04 (3.39e+06) | 1.38e+05 (4.58e+08) |
| balanced eq.(15) | 1.60e+02 (7.14e+02) | 2.66e+03 (1.38e+05) | 3.16e+04 (2.27e+07) | 3.65e+05 (3.24e+09) |
| n=128, r=100 | 8.21e+01 (4.74e+02) | 1.42e+03 (5.80e+04) | 1.09e+04 (3.35e+06) | 1.50e+05 (6.39e+08) |

Balanced steps of eq. (15): sig_L = 1.374, sig_m = 91.44, ratio sig_m/sig_L = 66.56.

Reference run, final certificates: spec (unprojected) = 3.275e+00, projected = 8.260e-14; J* h^2 = 3.658478765389.

## Target: quartic

Iterations to gap < threshold (None = not reached in 3000):
| run | sig_L | sig_m | <0.01 | <0.0001 | <1e-06 | <1e-08 | final gap | time (s) |
|---|---|---|---|---|---|---|---|---|
| scalar | 1.27e-05 | 1.27e-05 | None | None | None | None | 4.671e-01 | 5.0 |
| precond r=0.1 | 1 | 0.1 | 955 | 2968 | None | None | 9.422e-05 | 5.2 |
| precond r=1 | 1 | 1 | 99 | 300 | 578 | 888 | 4.441e-16 | 4.7 |
| precond r=10 | 1 | 10 | 14 | 38 | 61 | 92 | 1.332e-15 | 3.9 |
| precond r=100 | 1 | 100 | 8 | 23 | 40 | 58 | 1.332e-15 | 3.8 |
| precond (0.1, 1) | 0.1 | 1 | 86 | 296 | 574 | 885 | 1.332e-15 | 4.7 |
| precond (10, 100) | 10 | 100 | 40 | 206 | 375 | 545 | 1.332e-15 | 4.4 |
| balanced eq.(15) | 1.924 | 167.1 | 10 | 42 | 76 | 110 | 1.332e-15 | 3.9 |
| n=128, r=100 | 1 | 100 | 8 | 44 | 89 | 135 | 4.441e-16 | 12.8 |

Certificate / true gap at the first certificate evaluation (every 25 its) after each threshold is crossed — projected certificate (spec certificate in parentheses):
| run | <0.01 | <0.0001 | <1e-06 | <1e-08 |
|---|---|---|---|---|
| scalar | None | None | None | None |
| precond r=0.1 | 6.47e+00 (1.12e+02) | 2.02e+01 (1.47e+04) | None | None |
| precond r=1 | 1.59e+01 (1.11e+02) | 1.14e+02 (1.47e+04) | 1.12e+03 (2.22e+06) | 9.64e+03 (1.89e+08) |
| precond r=10 | 3.42e+02 (1.05e+03) | 6.28e+03 (2.05e+05) | 2.71e+04 (1.55e+07) | 1.31e+05 (5.15e+08) |
| precond r=100 | 2.89e+03 (3.08e+04) | 2.89e+03 (3.08e+04) | 7.64e+04 (2.40e+07) | 1.89e+06 (1.69e+10) |
| precond (0.1, 1) | 9.30e+00 (1.51e+02) | 2.22e+01 (1.57e+04) | 1.13e+02 (1.59e+06) | 1.01e+03 (1.97e+08) |
| precond (10, 100) | 2.04e+02 (3.14e+02) | 2.55e+03 (2.65e+04) | 1.90e+04 (1.60e+06) | 1.96e+05 (1.83e+08) |
| balanced eq.(15) | 5.96e+02 (1.76e+03) | 3.42e+03 (4.82e+04) | 9.96e+04 (4.56e+07) | 5.27e+05 (1.35e+09) |
| n=128, r=100 | 5.47e+02 (2.09e+03) | 2.13e+03 (2.34e+04) | 2.57e+04 (3.77e+06) | 2.99e+05 (5.75e+08) |

Balanced steps of eq. (15): sig_L = 1.924, sig_m = 167.1, ratio sig_m/sig_L = 86.85.

Reference run, final certificates: spec (unprojected) = 1.579e+00, projected = 5.684e-14; J* h^2 = 3.448541318441.

## Channel ratio

r_best = 100 (argmin of iterations to 1e-6 over sig_L = 1, sig_m = r, summed over the two targets); written to r_best.json and used by E3/E4.

## Deviation from the spec: the certificate

The certificate of Remark 4 as specified, `J - (-h1* - h2* - R ||B^T Lam + mu||_1)`, is a valid upper bound on the suboptimality but does **not** vanish: at a saddle point only `K2^T (B^T Lam + mu) = 0` holds, and `r = B^T Lam + mu` keeps a nonzero part orthogonal to gradient fields (see the 'spec' values above, which stay O(1)). Since `<K2 psi, r> = <K2 psi, P r>` with `P` the orthogonal projection onto range(K2) (one extra DCT pair), `R ||P r||_1` is an equally valid bound and it does vanish. All stopping rules (E3, E4) use this projected certificate; both are recorded here.

# Ensemble over 50 random instances

Seed = instance: Gaussian target with centre ~ U([0.35, 0.65]^2) and variances log-uniform on [0.01, 0.06]; field Y = x + 0.08 (grad phi + curl chi) with random cosine-series potentials (coefficients N(0,1)/|k|^3, |k|_inf <= 6, unit sup norm), so Y is not a gradient. All solver configurations run on the same instance (n = 64, N = 3000). Thresholds are on the gap normalised by the initial gap J(0) - J*. Iterations: median [95% bootstrap CI of the median]; a run that never reaches the threshold is right-censored (inf).

| configuration | its to 1e-4 | reached | its to 1e-8 | reached |
|---|---|---|---|---|
| scalar-step PDHG | inf [inf, inf] | 0/50 | inf [inf, inf] | 0/50 |
| preconditioned, \sigma_\mu/\sigma_\Lambda=0.1 | 1.5e+03 [1.36e+03, 1.66e+03] | 50/50 | inf [inf, inf] | 5/50 |
| preconditioned, \sigma_\mu/\sigma_\Lambda=1 | 153 [142, 161] | 50/50 | 424 [375, 476] | 50/50 |
| preconditioned, \sigma_\mu/\sigma_\Lambda=10 | 54 [53, 57] | 50/50 | 121 [111, 126] | 50/50 |
| preconditioned, \sigma_\mu/\sigma_\Lambda=100 | 52 [51, 56] | 50/50 | 128 [120, 135] | 50/50 |
| preconditioned (0.1, 1) | 161 [147, 178] | 50/50 | 428 [381, 485] | 50/50 |
| preconditioned (10, 100) | 498 [491, 533] | 50/50 | 1.22e+03 [1.16e+03, 1.28e+03] | 50/50 |
| balanced eq. (15) | 56 [51, 61] | 50/50 | 135 [123, 148] | 50/50 |

Projected certificate / true gap at the first certificate evaluation after the threshold is crossed (median [2.5, 97.5 percentile] over instances):

| configuration | at 1e-4 | at 1e-8 |
|---|---|---|
| scalar-step PDHG | — | — |
| preconditioned, \sigma_\mu/\sigma_\Lambda=0.1 | 3.12e+01 [1.8e+01, 6.0e+01] | 5.30e+03 [4.2e+03, 6.0e+03] |
| preconditioned, \sigma_\mu/\sigma_\Lambda=1 | 4.62e+02 [1.5e+02, 1.2e+03] | 2.10e+04 [8.4e+03, 7.3e+04] |
| preconditioned, \sigma_\mu/\sigma_\Lambda=10 | 3.95e+03 [1.6e+03, 2.6e+04] | 2.38e+05 [8.5e+04, 1.5e+06] |
| preconditioned, \sigma_\mu/\sigma_\Lambda=100 | 2.13e+03 [9.3e+02, 6.0e+03] | 1.41e+05 [5.4e+04, 7.3e+05] |
| preconditioned (0.1, 1) | 3.47e+01 [2.0e+01, 8.1e+01] | 2.44e+03 [9.3e+02, 7.5e+03] |
| preconditioned (10, 100) | 9.66e+02 [8.1e+02, 1.8e+03] | 8.51e+04 [4.5e+04, 1.7e+05] |
| balanced eq. (15) | 1.83e+03 [9.2e+02, 5.0e+03] | 1.62e+05 [6.3e+04, 5.2e+05] |

- balanced ratio sig_m/sig_L of eq. (15): median 41.8, range [24.1, 55.7]
- ratio with the smallest median iterations to 1e-6 over the ensemble: 10 (medians: r=0.1: 2857, r=1: 277, r=10: 86, r=100: 89)
- converged reference runs: certificate as written in Remark 4, median 1.377e+00 (min 9.095e-01); projected certificate, median 1.005e-13 (max 1.845e-13)
