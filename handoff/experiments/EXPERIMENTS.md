# Experiment specification — "Accelerated Proximal Optimization in Wasserstein Space with a Fixed Reference Measure"

This document is the complete specification for the experiments in Section 6 of the paper
(`paper/main.tex`, Section "Experiments", and Appendix D). It is written for a coding agent.
Read `paper/draft-v2.pdf` Sections 3–4 first (Algorithm 1 = outer scheme, Algorithm 2 = inner solver).

Environment: Python 3, numpy, scipy, matplotlib. No other packages are required. Use
`scipy.fft.dctn/idctn/dstn/idstn`. Everything is CPU and single-process. Use float64 everywhere.

Deliverables (put under `results/`): for every run a `.npz` with the raw curves, a `.pdf` figure,
and a `results.md` with the key numbers listed at the end of each experiment. Fix all seeds.
Every script must be re-runnable from the command line with no arguments (defaults = the spec).

---------------------------------------------------------------------------------------------------
## 0. Common definitions

### 0.1 Outer scheme (Algorithm 1 of the paper), maps as arrays
State: two "maps" X_t, Z_t (same array type as the map representation of the experiment).
```
alpha_t = 2/(t+2)                      # t = 0,1,2,...   (alpha_0 = 1)
Y_t     = (1-alpha_t) X_t + alpha_t Z_t
X_{t+1} = RESTRICTED_PROX(Y_t, gamma)  # argmin_{S in K_nu} G(S#nu) + ||S - Y_t||^2_nu/(2 gamma)
Z_{t+1} = Z_t + (X_{t+1} - Y_t)/alpha_t
```
Non-accelerated baseline ("minimizing movement"): same with `alpha_t = 1` for all t, i.e.
`X_{t+1} = RESTRICTED_PROX(X_t, gamma)`.
Initialization: `X_0 = Z_0 = Id` (so x_0 = nu).

### 0.2 Reference curves and the bound
Objective gap `gap_t = G(x_t) - G*`. Plot on log-log against t (outer steps, t >= 1) with:
- reference `C1/t` and `C2/t^2` lines fitted at t = 5 (same starting point as the curves),
- the bound of Theorem 1: `2 * W2nu0 / (gamma (t+1)^2)` where `W2nu0 = ||X_0 - P||^2_nu`
  (= W_2^2(nu, rho*) when X_0 = Id). For Gaussians compute it exactly (0.4); for grid experiments
  use `||Id - T||^2` with T the optimal map to the target obtained from the converged reference run.

### 0.3 Objective G = KL(. | q), gap floor
`G(rho) = ∫ rho log rho + ∫ V d rho + const`, `V = -log q`. Only differences matter; constants are
dropped consistently. `G*` for Gaussians is 0 (rho* = q). For grid experiments `G*` is the
objective value of a long high-accuracy reference run (see each experiment); floor plotted gaps at
1e-12.

### 0.4 Exact Gaussian machinery (d = 2, centered Gaussians; used in E1)
Covariances S (2x2 SPD). sqrt: `sqrtm_spd(S)` via eigh. Optimal map from N(0,A) to N(0,B):
```
T = A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}      # SPD, T A T = B
W2^2(A,B) = tr(A) + tr(B) - 2 tr( (A^{1/2} B A^{1/2})^{1/2} )
KL(S | Q) = 0.5 * ( tr(Q^{-1} S) - 2 - logdet(Q^{-1} S) )
```
Maps from the reference nu = N(0, Zc) (Zc fixed, A := Zc^{1/2}) are 2x2 matrices M acting as
x -> M x; `||M - M'||^2_nu = tr( (M - M') Zc (M - M')^T )`. Measure of a map: covariance `M Zc M^T`.
Optimal maps from nu are exactly the SPD matrices.
**Restricted prox for Gaussians** (the core of E1):
```
RESTRICTED_PROX_GAUSS(Y, gamma):   # Y any 2x2 real matrix
   minimize over SPD S:   KL(S Zc S | Q) + tr((S - Y) Zc (S - Y)^T) / (2 gamma)
```
Parametrize S = L L^T with L lower triangular (3 params, exp on the diagonal), minimize with
`scipy.optimize.minimize(method='BFGS')` with an analytic or finite-difference gradient, from the
starting point S = previous X_t (warm start), then polish with Nelder–Mead (xatol 1e-13, fatol 1e-15).
Verify: the KKT residual `|| grad_S ||` < 1e-10 (gradient of the objective w.r.t. the symmetric S,
projected onto symmetric matrices). This solve is a 3-parameter problem; it must be solved to
machine precision — the whole point of E1 is "exact steps".

**JKO step for Gaussians** (moving-reference scheme only, used as the comparison curve):
x+ = argmin_S KL(S|Q) + W2^2(S, Sy)/(2 gamma). Optimality: with M := I + gamma (Q^{-1} - S^{-1}),
M must be SPD and M S M = Sy. Solve the 3 equations of `M S M - Sy = 0` for S (Cholesky-param)
with `scipy.optimize.least_squares` (xtol=ftol=gtol=1e-15) from the starts Sy, Q, (Sy+Q)/2, and
accept the first solution with M SPD and residual < 1e-9*max(1,|Sy|_max). (For Q = c I there is the
closed form: eigenvalues u of S^{1/2} satisfy u = (sqrt(sy) + sqrt(sy + 4 gamma(1+gamma)))/(2(1+gamma))
... only for Q = I; do NOT rely on it for anisotropic Q.)
Moving-reference iteration (Section 5, eq. (16)), all 2x2 matrices, z_t has covariance Zt:
```
X_t = OTmap(Zt, Sx_t); Y = (1-a) X_t + a I; Sy = Y Zt Y^T
Sx_{t+1} = JKO(Sy, gamma); T = OTmap(Sy, Sx_{t+1}); X1 = T Y
Zt_{+} = (I + (X1 - Y)/a) Zt (I + (X1 - Y)/a)^T
```

### 0.5 Grid machinery (box Omega = [0,1]^2; used in E2–E4)
Grid: n x n cell centers `x_i = (i + 0.5)/n`. The potential perturbation psi is an n x n array
(cell values), defined modulo constants (always subtract the mean after each primal update).
**Cosine-series collocation.** Let `kap = pi * arange(n)` (per axis). Transforms (all with
`norm='ortho'`): C = DCT-II (dctn type 2), S = DST-II (dstn type 2). Represent psi by its
DCT-II coefficients `c = dctn(psi, type=2, norm='ortho')`. Then at the cell centers, EXACTLY:
```
psi_x  = -idstn( kap_x * c, type=2, axes=0 ) along x (DST-II inverse in x, DCT-II inverse in y)
psi_y  = -idstn( kap_y * c, type=2, axes=1 )           (DCT-II inverse in x, DST-II inverse in y)
psi_xx = -idctn( kap_x^2 * c )           psi_yy = -idctn( kap_y^2 * c )
psi_xy = +idstn( kap_x * kap_y * c, type=2, axes=(0,1) )   (DST-II inverse in both axes)
```
Implement a helper `apply_mixed_inverse(coef, sine_axes)` that applies the inverse DST-II on the
listed axes and inverse DCT-II on the others. NOTE: index k = 0 has kap = 0, so the DST component
k=0 is zero; scipy's DST-II of size n is indexed k = 1..n, so when you place a cosine coefficient
with index k into a sine array use position k-1 and drop k = 0 (it is multiplied by kap_0 = 0
anyway) — write a unit test that `psi_x` computed this way matches the analytic derivative of
`psi = cos(pi*3*x) cos(pi*2*y)` to 1e-10 at the cell centers, and that the adjoint identity
`<K psi, w> = <psi, K^T w>` holds to 1e-12 for random psi, w (define K^T by the transposed
transform sequence).
Operators (all on cell-center arrays, inner product = plain sum, i.e. unweighted; the factor h^2 is
dropped consistently everywhere, see below):
```
K2 psi = (psi_x, psi_y)                      shape (n,n,2)
K1 psi = (psi_xx, sqrt(2) psi_xy, psi_yy)    shape (n,n,3)     # sqrt(2): Frobenius inner product
K2^T K2 is diagonal in the DCT basis with multiplier  om(k)   = kap_kx^2 + kap_ky^2
K1^T K1 is diagonal in the DCT basis with multiplier  om(k)^2
```
Unit-test these two multiplier identities to 1e-10 on random psi (compare K^T K psi with
idctn(om * dctn(psi))).
**Weights.** All sums over cells carry the weight w(x) = nu(x) * h^2 (h = 1/n); for the uniform
reference w = h^2 is constant and we simply optimize J/h^2 (drop it) and multiply reported objective
values by h^2. For a non-uniform reference (E4 only) the weights are pointwise and are kept inside
h1 and h2 (see 0.6).
**The map** of psi is `S(x) = x + grad psi(x)`; it is admissible iff `I + Hess psi(x)` is positive
definite at every cell (check: a + 1 > 0, c + 1 > 0, (a+1)(c+1) - b^2 > 0 with Hess = [[a,b],[b,c]]).
**Objective of the proximal step** (eq. (11)), for a given field Y (n,n,2) and gamma:
```
J(psi) = sum_cells w(x) * [ -log det(I + Hess psi) + V(x + grad psi) + |x + grad psi - Y|^2/(2 gamma) ]
```
(+inf if not admissible). `G(x_t)` for reporting = `sum w [ -log det(I + Hess psi_t) + V(x + grad psi_t) ]`
(entropy of the reference dropped; for non-uniform nu add `sum w log nu`, a constant).

### 0.6 Inner solver (Algorithm 2 of the paper): two-channel spectrally preconditioned PDHG
Dual variables: Lam (n,n,3) for the entropy channel (symmetric 2x2 per cell stored as
(l11, sqrt(2) l12, l22) so that the Frobenius inner product is the plain dot product), mu (n,n,2)
for the transport channel. Steps sig_L, sig_m > 0, theta in (0,1) (use theta = 0.99).
Preconditioner on the primal: in the DCT basis `Tk = theta / (sig_L * om^2 + sig_m * om)` for
k != 0 and `Tk = 0` at k = 0.
```
psi = 0; psib = 0; Lam = 0; mu = 0
for k in range(N):
    H  = K1 psib ; g = K2 psib
    Lam = prox_{sig_L h1*}(Lam + sig_L H)        # pointwise, see below
    mu  = prox_{sig_m h2*}(mu  + sig_m g)        # pointwise, see below
    grad = K1^T Lam + K2^T mu                     # n x n
    psi_new = psi - idctn( Tk * dctn(grad) );  psi_new -= mean(psi_new)
    psib = 2 psi_new - psi ; psi = psi_new
```
**Entropy channel prox** (h1(H) = w * (-log det(I+H)), pointwise; Moreau):
```
prox_{s h1*}(L) = L - s * prox_{h1/s}(L/s)
prox_{t h1}(M):  symmetrize M = [[m11, m12],[m12, m22]] (from the 3-vector, m12 = v2/sqrt2);
                 eigen-decompose M = U diag(e1,e2) U^T;  with tw = t * w(x):
                 p_i = ( (e_i + 1) + sqrt((e_i + 1)^2 + 4 tw) ) / 2  - 1
                 return U diag(p1,p2) U^T  (back to 3-vector with sqrt2 on the off-diagonal)
```
(This is the minimizer of -tw log det(I+P) + 0.5 |P - M|^2_F; verify by a finite-difference test.)
**Transport channel prox** (h2(m) = w * [ V(x+m) + |x+m-Y|^2/(2 gamma) ], pointwise; Moreau):
```
prox_{s h2*}(v) = v - s * prox_{h2/s}(v/s)
prox_{t h2}(m0): argmin_m  w*[ V(x+m) + |x+m-Y|^2/(2gamma) ] + |m - m0|^2/(2t)
```
For SEPARABLE V (all targets in E2–E3 are separable, see below) this splits per coordinate into a
1-D strictly convex problem on the feasible interval `x_i + m_i in [0,1]` (the target is supported
in the box, V = +inf outside): solve by safeguarded Newton (10 iterations, then bisection fallback)
on that interval; for quadratic V it is a clamped closed form:
`m_i = clip( (t w (Y_i - x_i)/gamma + t w Qinv_i (q_i - x_i) ... ) )` — simplest is to just use the
generic 1-D Newton with the quadratic; do not special-case.
**Certificate** (Remark 4 of the paper), evaluated every 25 iterations and at the end:
```
h1*(Lam) = sum_cells w * ( -tr(Lam/w) ... )
```
Derive it cleanly: with h1(H) = w f(H), f(H) = -log det(I+H), the conjugate is
h1*(L) = w f*(L/w), f*(Lt) = -tr(Lt) - 2 - log det(-Lt) for Lt negative definite (else +inf).
Likewise h2*(v) = w g*(v/w) with g(m) = V(x+m)+|x+m-Y|^2/(2gamma) + indicator(x+m in box), and
g*(u) = sup_m <m,u> - g(m): a separable 1-D concave maximization per coordinate on the interval
(golden-section or Newton, 1e-10). Then
```
B^T Lam + mu  =  K2-adjoint-free form: since K1 = B K2 with B(g1,g2) = ( d_x g1, (1/sqrt2)(d_y g1 + d_x g2), d_y g2 ),
                 B^T Lam = ( d_x^T L1 + (1/sqrt2) d_y^T L2 ,  (1/sqrt2) d_x^T L2 + d_y^T L3 )   (spectral adjoints)
lower_bound = -h1*(Lam) - h2*(mu) - R * || B^T Lam + mu ||_1      with R = 1 (|grad psi_i| <= 1 for feasible maps)
certificate eps = J(psi_feasible) - lower_bound                    (>= true suboptimality)
```
Unit-test weak duality: on a converged run, lower_bound <= J(psi) and the certificate -> 0.
If h1*/h2* is +inf for the current duals (should not happen, the proxes keep duals in the domain),
report inf and continue.
**Output map** of the inner solver: the last primal iterate psi that is admissible (I + Hess > 0 at
all cells). Track `last_admissible`.
**Scalar-step baseline** (for E2 only): standard Chambolle–Pock with T = tau I, sig_L = sig_m = sigma,
tau = sigma = 1/||K|| where ||K||^2 = max_k (om^2 + om) (theta = 0.99 applied as tau*sigma*||K||^2 = 0.99).

### 0.7 Targets (all supported in the box, V = -log q + const, separable except E4)
- GAUSS(m, s1, s2):  V(u) = (u1-m1)^2/(2 s1^2) + (u2-m2)^2/(2 s2^2)  on the box, +inf outside.
- QUARTIC(m, c):     V(u) = c * [ (u1-m1)^4 + (u2-m2)^4 ] / 4           on the box, +inf outside.
- MIX2(w, m1, m2, s): q(u) ∝ w N(u; m1, s^2 I) + (1-w) N(u; m2, s^2 I) restricted to the box; V = -log q.
  Non-separable and non-convex (E4 only): the transport prox is a 2-D problem per cell; use 5 Newton
  steps from the previous value with backtracking, projected to the box.

### 0.8 Plot style
matplotlib, log-log for gap-vs-t, one panel per setting, methods as solid lines, references dashed
grey, the bound as a dotted line. Label everything. Save PDF and PNG. Use seeds 0.

---------------------------------------------------------------------------------------------------
## E2 — Inner solver (run FIRST: fastest, and it fixes the channel ratio used later)
Script: `e2_inner_solver.py`. Grid n = 64 (and n = 128 for one scaling run). gamma = 0.05.
Reference nu = uniform. Field Y := x + 0.08 * grad( sin(2 pi x) cos(4 pi y) + 0.5 cos(2 pi y) )
(a generic non-gradient-free field; compute the gradient analytically). Targets:
  (a) GAUSS(m=(0.5,0.5), s1=sqrt(0.02), s2=sqrt(0.05));  (b) QUARTIC(m=(0.5,0.5), c=200).
Reference optimum J*: run the preconditioned solver with ratio 10 for 20,000 iterations and take
the minimum J over the run (store psi*).
Runs (each N = 3000 iterations, record J(psi_k) - J*, the certificate every 25 its, wall-clock):
  1. scalar-step Chambolle–Pock (0.6 baseline)
  2. preconditioned, sig_L = 1, sig_m = r for r in {0.1, 1, 10, 100}
  3. preconditioned, both scaled: (sig_L, sig_m) in {(0.1, 1), (10, 100)}
  4. the "balanced" choice of eq. (15): sig_L = sqrt(theta) ||Lam*||/||K1 psi*||, sig_m = sqrt(theta) ||mu*||/||K2 psi*||
     using the converged duals/primal from the reference run (report the resulting values and ratio)
  5. n = 128 with the best ratio from 2 (expect iteration counts roughly independent of n)
Report in results.md: iterations to gap < 1e-2, 1e-4, 1e-6, 1e-8 for every run (None if not reached),
the ratio of certificate to true gap at those points, and the value of the balanced steps. Figure
E2: (left) gap vs iteration for scalar vs preconditioned r in {1, 10, 100}, both targets as two
rows; (right) certificate vs true gap for r = 10.
Expected (from a torus prototype): scalar never reaches 1e-2 in 3000; r = 1: 1e-8 at ~460;
r = 10: 1e-8 at ~260; r = 0.1 much slower. If the preconditioned solver diverges, first suspect the
DST index shift (0.5) and the adjoints; the unit tests must pass before running anything.
Choose `r_best` = argmin iterations-to-1e-6 over run 2; it is used in E3/E4.

## E1 — Rates with exact steps (Gaussians)  [fast; pure 2x2 linear algebra]
Script: `e1_gaussian_rates.py`. d = 2, centered. Reference nu = N(0, Zc) with
Zc = R(pi/5) diag(6.25, 2.25) R(pi/5)^T (R = rotation). gamma = 0.5. T = 300 outer steps.
Targets: WEAK: Q = diag(25, 10) (precisions 0.04, 0.1); STRONG: Q = diag(1, 0.5).
Methods: (i) Algorithm 1 (accelerated, fixed reference, exact restricted prox of 0.4);
(ii) fixed-reference minimizing movement (alpha = 1); (iii) moving-reference scheme (0.4, JKO solver);
(iv) plain JKO (moving reference without momentum: x_{t+1} = JKO(x_t)), as the classical baseline.
Record gap_t = KL(Sx_t | Q) for all t, and for (i) also ||Z_t - P||_nu (P = OTmap(Zc, Q)) to confirm
boundedness. Plot: Figure E1 with two panels (WEAK, STRONG), 4 methods, references 1/t and 1/t^2,
and the bound 2 ||I - P||^2_nu / (gamma (t+1)^2) for (i).
Lambda sweep (Figure E1b): Q = (1/lam) I for lam in logspace(-3, 0, 13), N = 200 steps, final gap
for (i) and (ii) and (iv); plot final gap vs lam (log-log). Expect (i) nearly flat, (ii)/(iv) growing
as lam -> 0.
Sanity: for (i) the gap must be below the bound at every t (if not, the restricted prox is not
solved accurately enough — tighten the solver, do not touch the algorithm).
results.md: gap at t = 10, 100, 300 for every method/target; max_t gap/bound for (i); max_t ||Z_t-P||/||I-P||.

## E3 — End to end (outer scheme with the inner solver)  [moderate; run after E2]
Script: `e3_end_to_end.py`. Grid n = 64, nu uniform, X_0 = Id (psi = 0, Z_0 = x), gamma = 0.05,
T = 60 outer steps. Targets: GAUSS(a) and QUARTIC(b) of E2. Inner solver: preconditioned PDHG with
(sig_L, sig_m) = (1, r_best), warm-started from the previous psi (and previous duals), stopped when the
certificate <= eps_t with eps_t = eps0 (t+1)^{-4.5}, eps0 = 1e-2 (in the h^2-weighted objective units),
hard cap 5000 inner iterations (record if hit). Reference G*: fixed-reference minimizing movement for
1500 outer steps at certificate 1e-10 (cap 20000); take G* = min G over that run.
Runs: (i) accelerated + certified schedule; (ii) non-accelerated + certified schedule;
(iii) accelerated + fixed 50 inner iterations per step; (iv) non-accelerated + fixed 50.
Record per outer step: gap, inner iterations used, cumulative inner iterations, wall-clock.
Figure E3: (left) gap vs outer step t for (i)-(iv) with 1/t and 1/t^2 references; (right) gap vs
cumulative inner iterations. results.md: gap at t = 10, 30, 60; total inner iterations; whether the
cap was ever hit; the empirical slope of (i) between t = 10 and 60.

## E4 — Beyond the theory (non-log-concave target)  [slowest; run last, can be cut short]
Script: `e4_mixture.py`. Grid n = 64, gamma = 0.05, T = 80 outer steps, inner solver as in E3 (certified
schedule, cap 2000; the inner problem is non-convex here, document that and just run it).
Target: MIX2(w, m1 = (0.3, 0.5), m2 = (0.7, 0.5), s = 0.06) with w in {0.5, 0.9}.
Three reference measures (= three initializations, since x_0 = nu):
  (A) uniform on the box; (B) truncated N((0.5,0.5), 0.15^2 I) (density on the grid, normalized);
  (C) truncated N((0.52,0.5), 0.15^2 I)  (slightly asymmetric).
For non-uniform nu the weights w(x) = nu(x_cell) h^2 enter h1, h2 and the objective as in 0.5/0.6.
Methods: accelerated and non-accelerated fixed-reference. Record gap (G* = best value seen across all
runs for that target, documented as such), and save the pushed-forward cell centers S(x_cells) with
density 1/det(I+Hess psi) at t in {0, 5, 10, 20, 40, 80}.
Figure E4: for w = 0.9, init (C): 2 x 6 scatter panels (accelerated / non-accelerated, times above),
points colored by density; plus one gap-vs-t panel for all 2 targets x 3 inits (12 curves, two
colors by method, line styles by init). results.md: gaps at t = 20, 80; whether either method
collapsed to one mode (fraction of mass in the left half-plane at t = 80 vs the target's w).

---------------------------------------------------------------------------------------------------
## Order of runs and rough budget (single core)
1. Unit tests (transforms, adjoints, multipliers, proxes, weak duality): minutes. Do not proceed
   until all pass.
2. E2: ~ 10 runs x 3000 its x ~3 ms (n=64) ≈ 2 min; n=128 run ≈ 5 min; reference 20k its ≈ 1 min each target.
3. E1: seconds to a few minutes (the restricted prox is a 3-parameter solve; 300 steps x 4 methods x 2 targets, plus the sweep).
4. E3: 4 runs x 60 outer x up to 5000 inner at ~3 ms ≈ up to 1 h worst case; typically 10–20 min. Reference run ≤ 1 h.
5. E4: 12 runs x 80 outer x up to 2000 inner ≈ up to 1–2 h.
Total: well within a night. Write results.md incrementally so partial results survive.

## What to send back
`results/` (npz, pdf, png, results.md), the scripts, and the unit-test output. Note any deviation
from this spec explicitly.
