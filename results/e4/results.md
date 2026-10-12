# E4 — beyond the theory (two-component mixture)

n = 64, gamma = 0.05, T = 80, target MIX2(w, (0.3,0.5), (0.7,0.5), s = 0.06), inner solver (sig_L, sig_m) = (1, 100), schedule eps_t = 0.01 (t+1)^-4.5, cap 2000; an inner solve also ends ('stalled') once its coefficients move by less than 1e-10 (relative) over 25 iterations.

**Outside the theory.** G is not convex along generalized geodesics, the inner problem is non-convex (the pointwise transport solve is a warm-started local Newton solve, and the 'certificate' uses a local maximisation for h2*, so it is not a valid bound). No scheme has a guarantee here. G = KL(x | q) with q normalised on the box; G* = the best value seen across all runs for that target.

## w = 0.5

G* (best value seen) = 0.01555331. Target mass in the left half-plane: 0.5.

| reference / init | method | gap at t=20 | gap at t=80 | left mass at t=80 | inner its | steps at cap | steps stalled | steps w/o admissible iterate |
|---|---|---|---|---|---|---|---|---|
| A (uniform) | accelerated | 2.034e-01 | 2.034e-01 | 0.500 | 4100 | 0 | 80 | 0 |
| A (uniform) | non-accelerated | 2.034e-01 | 2.034e-01 | 0.500 | 3600 | 0 | 80 | 0 |
| B (trunc. N((0.5,0.5), 0.15^2 I)) | accelerated | 4.726e-07 | 1.221e-15 | 0.500 | 130250 | 64 | 16 | 0 |
| B (trunc. N((0.5,0.5), 0.15^2 I)) | non-accelerated | 4.744e-07 | 0.000e+00 | 0.500 | 130400 | 64 | 16 | 0 |
| C (trunc. N((0.52,0.5), 0.15^2 I)) | accelerated | 6.181e-03 | 6.204e-03 | 0.447 | 160000 | 80 | 0 | 0 |
| C (trunc. N((0.52,0.5), 0.15^2 I)) | non-accelerated | 6.181e-03 | 6.204e-03 | 0.447 | 160000 | 80 | 0 | 0 |

## w = 0.9

G* (best value seen) = 0.28627512. Target mass in the left half-plane: 0.9.

| reference / init | method | gap at t=20 | gap at t=80 | left mass at t=80 | inner its | steps at cap | steps stalled | steps w/o admissible iterate |
|---|---|---|---|---|---|---|---|---|
| A (uniform) | accelerated | 6.197e-02 | 6.197e-02 | 0.719 | 7875 | 0 | 80 | 0 |
| A (uniform) | non-accelerated | 1.497e-01 | 1.497e-01 | 0.656 | 5700 | 0 | 80 | 0 |
| B (trunc. N((0.5,0.5), 0.15^2 I)) | accelerated | 4.743e-07 | 4.896e-12 | 0.623 | 160000 | 80 | 0 | 0 |
| B (trunc. N((0.5,0.5), 0.15^2 I)) | non-accelerated | 6.626e-07 | 1.868e-07 | 0.623 | 160000 | 80 | 0 | 0 |
| C (trunc. N((0.52,0.5), 0.15^2 I)) | accelerated | 9.298e-02 | 9.298e-02 | 0.571 | 160000 | 80 | 0 | 0 |
| C (trunc. N((0.52,0.5), 0.15^2 I)) | non-accelerated | 9.298e-02 | 9.298e-02 | 0.571 | 160000 | 80 | 0 | 0 |

Collapse check: a run has collapsed to one mode if its left mass is near 0 or 1 rather than near w.

# Ensemble over 20 random instances

Seed = instance: mixture weight w ~ U(0.6, 0.95), modes at (0.5 -+ delta, 0.5) with delta ~ U(0.15, 0.25), s = 0.06; references: A = uniform reference; C = trunc. N((0.5 + off, 0.5), 0.15^2 I), off ~ N(0, 0.02^2). Both methods and both references run on the same instance; G* = best value seen across the four runs of that instance. Mode recovery = mass in the left half-plane at t = 80 within +-0.05 of w (Wilson 95% interval). Gap columns: normalised gap (G(x_t) - G*)/(G(x_0) - G*), mean [95% bootstrap CI of the mean].

| reference | method | mode recovery rate | abs(left mass - w) | normalised gap at t=20 | normalised gap at t=80 | steps w/o admissible iterate |
|---|---|---|---|---|---|---|
| A | accelerated | 0/20 [0.00, 0.16] | 0.173 [0.145, 0.203] | 0.0162 [0.0132, 0.0187] | 0.0162 [0.0132, 0.0187] | 0 |
| A | non-accelerated | 0/20 [0.00, 0.16] | 0.178 [0.153, 0.207] | 0.017 [0.0145, 0.0192] | 0.0168 [0.0143, 0.0189] | 0 |
| C | accelerated | 0/20 [0.00, 0.16] | 0.18 [0.149, 0.211] | 0.00383 [0.000161, 0.00868] | 0.00312 [0.000101, 0.00782] | 0 |
| C | non-accelerated | 0/20 [0.00, 0.16] | 0.187 [0.158, 0.216] | 0.00705 [0.00213, 0.0133] | 0.00533 [0.0011, 0.0108] | 0 |

Paired log10(gap accelerated / gap non-accelerated), gaps floored at 1e-13 (negative = acceleration helps):

| reference | t=20 | t=80 |
|---|---|---|
| A | -1.18 [-2.92, -0.009] | -1.18 [-2.92, -0.00563] |
| C | -0.336 [-0.879, 0.0164] | -0.628 [-1.74, 0.247] |
