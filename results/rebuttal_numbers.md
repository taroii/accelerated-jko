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


### e7_defect2d
NUMBER: e7_defect2d.delta_t_normalized_max = 1.79264
NUMBER: e7_defect2d.delta_t_normalized_mean = 1.16124
NUMBER: e7_defect2d.accumulated_defect_term = 16.0011
NUMBER: e7_defect2d.blocks = 25
NUMBER: e7_defect2d.particles = 12000
NUMBER: e7_defect2d.n_sub = 2048

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
