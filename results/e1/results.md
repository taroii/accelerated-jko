# E1 — rates with exact steps (Gaussians)

d = 2, nu = N(0, Zc), Zc = R(pi/5) diag(6.25, 2.25) R(pi/5)^T, gamma = 0.5, T = 300. gap_t = KL(x_t | q), G* = 0.

## WEAK: Q = diag(25, 10)

| method | gap at t=10 | gap at t=100 | gap at t=300 |
|---|---|---|---|
| Alg. 1 (accelerated, fixed reference) | 4.142e-02 | 8.551e-08 | 2.089e-11 |
| fixed-reference minimizing movement (\alpha_t\equiv1) | 1.487e-01 | 7.305e-05 | 1.445e-11 |
| moving-reference scheme | 4.114e-02 | 2.196e-07 | 1.153e-11 |
| plain JKO | 1.487e-01 | 6.642e-05 | 1.010e-11 |

- W_2^2(nu, rho*) = ||I - P||_nu^2 = 10.237052
- max_t gap/bound for Alg. 1: 0.2028 (must be <= 1)
- max_t ||Z_t - P||_nu / ||I - P||_nu: 1.0000 (Theorem 1: <= 1)
- max KKT residual of the restricted prox: accelerated 2.7e-15, minimizing movement 2.6e-15
- max residual of the JKO solves: moving reference 1.8e-14, plain JKO 1.1e-14

## STRONG: Q = diag(1, 0.5)

| method | gap at t=10 | gap at t=100 | gap at t=300 |
|---|---|---|---|
| Alg. 1 (accelerated, fixed reference) | 5.347e-06 | 2.465e-32 | -1.110e-16 |
| fixed-reference minimizing movement (\alpha_t\equiv1) | 2.616e-06 | -1.110e-16 | -1.110e-16 |
| moving-reference scheme | 5.376e-06 | 1.110e-16 | 0.000e+00 |
| plain JKO | 2.309e-06 | 1.110e-16 | 1.110e-16 |

- W_2^2(nu, rho*) = ||I - P||_nu^2 = 3.075459
- max_t gap/bound for Alg. 1: 0.2592 (must be <= 1)
- max_t ||Z_t - P||_nu / ||I - P||_nu: 1.0000 (Theorem 1: <= 1)
- max KKT residual of the restricted prox: accelerated 9.3e-16, minimizing movement 1.0e-15
- max residual of the JKO solves: moving reference 6.7e-16, plain JKO 8.9e-16

## Lambda sweep: Q = (1/lam) I, gap after 200 steps

| lam | Alg. 1 | min. movement | plain JKO |
|---|---|---|---|
| 1.000e-03 | 1.990e-04 | 8.774e-01 | 8.774e-01 |
| 1.778e-03 | 1.424e-03 | 4.966e-01 | 4.966e-01 |
| 3.162e-03 | 5.065e-04 | 2.208e-01 | 2.208e-01 |
| 5.623e-03 | 9.953e-05 | 6.545e-02 | 6.545e-02 |
| 1.000e-02 | 1.036e-05 | 9.551e-03 | 9.551e-03 |
| 1.778e-02 | 1.406e-06 | 3.868e-04 | 3.868e-04 |
| 3.162e-02 | 2.336e-09 | 1.522e-06 | 1.522e-06 |
| 5.623e-02 | 1.098e-10 | 9.690e-11 | 9.690e-11 |
| 1.000e-01 | 9.310e-15 | 9.652e-18 | -4.586e-17 |
| 1.778e-01 | 1.032e-21 | 5.551e-17 | 3.550e-30 |
| 3.162e-01 | 1.110e-16 | 1.972e-31 | 1.972e-31 |
| 5.623e-01 | -5.551e-17 | 0.000e+00 | 4.930e-32 |
| 1.000e+00 | 0.000e+00 | 0.000e+00 | 1.972e-31 |

# Ensemble over 100 random instances

Seed = instance: Zc = R(theta) diag(e^a1, e^a2) R(theta)^T with theta ~ U(0, pi), a_i ~ U(0, log 10); Q = R(theta') diag(1/lam1, 1/lam2) R(theta')^T with lam_i log-uniform on [0.02, 0.2] (weak) or [0.5, 5] (strong). All four methods run on the same instance. Reported quantity: the bound utilisation u_t = gap_t gamma (t+1)^2 / (2 W_{2,nu}^2(x_0, rho*)); Theorem 1 says u_t <= 1 for Alg. 1 on every instance. Tables: mean [95% bootstrap CI of the mean, 2000 resamples]; figures: median and 2.5-97.5 percentile band. Gaps are floored at 1e-14 before taking log-ratios.

## weak regime

| method | u at t=10 | u at t=100 | u at t=300 |
|---|---|---|---|
| Alg. 1 (accelerated, fixed reference) | 0.114 [0.102, 0.126] | 0.000422 [0.000293, 0.000571] | 1.2e-06 [6.4e-07, 1.83e-06] |
| fixed-reference minimizing movement (\alpha_t\equiv1) | 0.399 [0.387, 0.41] | 0.084 [0.0632, 0.107] | 8.8e-05 [5.21e-05, 0.000131] |
| moving-reference scheme | 0.114 [0.102, 0.126] | 0.000424 [0.000297, 0.000574] | 1.16e-06 [6.42e-07, 1.76e-06] |
| plain JKO | 0.399 [0.387, 0.409] | 0.0828 [0.0623, 0.105] | 8.41e-05 [4.96e-05, 0.000125] |

- Alg. 1: max over instances and t of u_t = 0.2522; instances with u_t <= 1 for all t: 100/100 (Wilson 95%: [0.963, 1.000])
- max over instances and t of ||Z_t - P||_nu / ||I - P||_nu: 1.000000
- paired log10(gap accelerated / gap minimizing movement): t=10: -0.693 [-0.784, -0.612]; t=100: -2.05 [-2.26, -1.83]; t=300: -0.766 [-1, -0.543]
- max KKT residual of the restricted prox over all instances: 5.2e-15
- instances in which a solve failed (run truncated): accel 0, mm 0, moving 0, jko 0

## strong regime

| method | u at t=10 | u at t=100 | u at t=300 |
|---|---|---|---|
| Alg. 1 (accelerated, fixed reference) | 3.93e-05 [2.6e-05, 5.44e-05] | 3.84e-14 [1.55e-14, 7.26e-14] | 1.07e-13 [5.02e-14, 1.76e-13] |
| fixed-reference minimizing movement (\alpha_t\equiv1) | 0.000352 [0.000214, 0.000512] | 5.33e-14 [2.65e-14, 8.6e-14] | 4.73e-13 [2.36e-13, 7.64e-13] |
| moving-reference scheme | 3.86e-05 [2.55e-05, 5.34e-05] | 6.11e-14 [2.19e-14, 1.23e-13] | 1.16e-12 [2.82e-13, 2.36e-12] |
| plain JKO | 0.000346 [0.000209, 0.000504] | 6.18e-14 [1.76e-14, 1.37e-13] | 5.49e-13 [1.56e-13, 1.21e-12] |

- Alg. 1: max over instances and t of u_t = 0.2690; instances with u_t <= 1 for all t: 100/100 (Wilson 95%: [0.963, 1.000])
- max over instances and t of ||Z_t - P||_nu / ||I - P||_nu: 1.000000
- paired log10(gap accelerated / gap minimizing movement): t=10: 0.131 [-0.0944, 0.363]; t=100: 0 [0, 0]; t=300: 0 [0, 0]
- max KKT residual of the restricted prox over all instances: 2.5e-15
- instances in which a solve failed (run truncated): accel 0, mm 0, moving 0, jko 0

## Lambda sweep, 50 random references per lambda: u after 200 steps (median [2.5, 97.5 percentile])

| lam | Alg. 1 (accelerated, fixed reference) | fixed-reference minimizing movement (\alpha_t\equiv1) | plain JKO |
|---|---|---|---|
| 1.000e-03 | 1.13e-03 [1.1e-03, 1.3e-03] | 5.02e+00 [4.8e+00, 5.3e+00] | 5.02e+00 [4.8e+00, 5.3e+00] |
| 1.778e-03 | 1.51e-02 [1.4e-02, 1.7e-02] | 5.28e+00 [5.0e+00, 5.6e+00] | 5.28e+00 [5.0e+00, 5.6e+00] |
| 3.162e-03 | 1.01e-02 [9.3e-03, 1.1e-02] | 4.43e+00 [4.1e+00, 4.8e+00] | 4.43e+00 [4.1e+00, 4.8e+00] |
| 5.623e-03 | 3.84e-03 [3.4e-03, 4.6e-03] | 2.54e+00 [2.3e+00, 2.9e+00] | 2.54e+00 [2.3e+00, 2.9e+00] |
| 1.000e-02 | 8.04e-04 [7.4e-04, 8.8e-04] | 7.38e-01 [6.6e-01, 8.7e-01] | 7.38e-01 [6.6e-01, 8.7e-01] |
| 1.778e-02 | 2.29e-04 [2.1e-04, 2.6e-04] | 6.26e-02 [5.5e-02, 7.7e-02] | 6.26e-02 [5.5e-02, 7.7e-02] |
| 3.162e-02 | 9.03e-07 [4.9e-07, 1.1e-06] | 5.58e-04 [4.8e-04, 7.3e-04] | 5.58e-04 [4.8e-04, 7.3e-04] |
| 5.623e-02 | 1.02e-07 [8.2e-08, 1.6e-07] | 9.09e-08 [7.7e-08, 1.3e-07] | 9.09e-08 [7.7e-08, 1.3e-07] |
| 1.000e-01 | 2.87e-11 [2.6e-11, 3.4e-11] | 3.21e-14 [0.0e+00, 3.1e-13] | 3.05e-14 [0.0e+00, 4.0e-13] |
| 1.778e-01 | 1.44e-17 [0.0e+00, 4.0e-12] | 8.59e-26 [0.0e+00, 2.6e-12] | 6.49e-26 [0.0e+00, 7.5e-13] |
| 3.162e-01 | 5.35e-27 [0.0e+00, 2.3e-12] | 6.28e-28 [0.0e+00, 1.7e-12] | 9.68e-28 [0.0e+00, 4.1e-26] |
| 5.623e-01 | 5.44e-28 [0.0e+00, 7.7e-12] | 1.85e-28 [0.0e+00, 9.8e-12] | 3.72e-28 [0.0e+00, 3.5e-26] |
| 1.000e+00 | 0.00e+00 [0.0e+00, 8.3e-13] | 0.00e+00 [0.0e+00, 3.4e-13] | 8.76e-28 [2.8e-28, 3.4e-26] |
