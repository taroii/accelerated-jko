# E4 — beyond the theory (two-component mixture)

n = 64, gamma = 0.05, T = 80, target MIX2(w, (0.3,0.5), (0.7,0.5), s = 0.06), inner solver (sig_L, sig_m) = (1, 100), schedule eps_t = 0.01 (t+1)^-4.5, cap 2000.

**Outside the theory.** G is not convex along generalized geodesics, the inner problem is non-convex (the pointwise transport solve is a warm-started local Newton solve, and the 'certificate' uses a local maximisation for h2*, so it is not a valid bound). No scheme has a guarantee here. G = KL(x | q) with q normalised on the box; G* = the best value seen across all runs for that target.

## w = 0.5

G* (best value seen) = 0.01555331. Target mass in the left half-plane: 0.5.

| reference / init | method | gap at t=20 | gap at t=80 | left mass at t=80 | inner its | steps at cap | steps w/o admissible iterate |
|---|---|---|---|---|---|---|---|
| A (uniform) | accelerated | 2.034e-01 | 2.034e-01 | 0.500 | 160000 | 80 | 0 |
| A (uniform) | non-accelerated | 2.034e-01 | 2.034e-01 | 0.500 | 160000 | 80 | 0 |
| B (trunc. N((0.5,0.5), 0.15^2 I)) | accelerated | 4.726e-07 | 0.000e+00 | 0.500 | 160000 | 80 | 0 |
| B (trunc. N((0.5,0.5), 0.15^2 I)) | non-accelerated | 4.744e-07 | 0.000e+00 | 0.500 | 160000 | 80 | 0 |
| C (trunc. N((0.52,0.5), 0.15^2 I)) | accelerated | 6.181e-03 | 6.204e-03 | 0.447 | 160000 | 80 | 0 |
| C (trunc. N((0.52,0.5), 0.15^2 I)) | non-accelerated | 6.181e-03 | 6.204e-03 | 0.447 | 160000 | 80 | 0 |

## w = 0.9

G* (best value seen) = 0.28627512. Target mass in the left half-plane: 0.9.

| reference / init | method | gap at t=20 | gap at t=80 | left mass at t=80 | inner its | steps at cap | steps w/o admissible iterate |
|---|---|---|---|---|---|---|---|
| A (uniform) | accelerated | 6.197e-02 | 6.197e-02 | 0.719 | 160000 | 80 | 0 |
| A (uniform) | non-accelerated | 1.497e-01 | 1.497e-01 | 0.656 | 160000 | 80 | 0 |
| B (trunc. N((0.5,0.5), 0.15^2 I)) | accelerated | 4.743e-07 | 4.896e-12 | 0.623 | 160000 | 80 | 0 |
| B (trunc. N((0.5,0.5), 0.15^2 I)) | non-accelerated | 6.626e-07 | 1.868e-07 | 0.623 | 160000 | 80 | 0 |
| C (trunc. N((0.52,0.5), 0.15^2 I)) | accelerated | 9.298e-02 | 9.298e-02 | 0.571 | 160000 | 80 | 0 |
| C (trunc. N((0.52,0.5), 0.15^2 I)) | non-accelerated | 9.298e-02 | 9.298e-02 | 0.571 | 160000 | 80 | 0 |

Collapse check: a run has collapsed to one mode if its left mass is near 0 or 1 rather than near w.
