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
- max KKT residual of the restricted prox: accelerated 2.5e-15, minimizing movement 2.7e-15
- max residual of the JKO solves: moving reference 1.8e-14, plain JKO 1.4e-14

## STRONG: Q = diag(1, 0.5)

| method | gap at t=10 | gap at t=100 | gap at t=300 |
|---|---|---|---|
| Alg. 1 (accelerated, fixed reference) | 5.347e-06 | 5.551e-17 | 5.551e-17 |
| fixed-reference minimizing movement (\alpha_t\equiv1) | 2.616e-06 | 5.551e-17 | 5.551e-17 |
| moving-reference scheme | 5.376e-06 | 4.930e-32 | 2.465e-32 |
| plain JKO | 2.309e-06 | 1.110e-16 | 1.110e-16 |

- W_2^2(nu, rho*) = ||I - P||_nu^2 = 3.075459
- max_t gap/bound for Alg. 1: 0.2592 (must be <= 1)
- max_t ||Z_t - P||_nu / ||I - P||_nu: 1.0000 (Theorem 1: <= 1)
- max KKT residual of the restricted prox: accelerated 1.1e-15, minimizing movement 1.0e-15
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
| 1.000e-01 | 9.216e-15 | -5.063e-17 | -5.063e-17 |
| 1.778e-01 | 1.110e-16 | 5.551e-17 | 1.972e-30 |
| 3.162e-01 | -1.110e-16 | 1.972e-31 | 1.972e-31 |
| 5.623e-01 | -5.551e-17 | 9.861e-32 | 2.465e-32 |
| 1.000e+00 | 0.000e+00 | 0.000e+00 | 9.861e-32 |
