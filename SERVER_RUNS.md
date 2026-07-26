# Server runs (full scale)

CPU experiments (E1–E6, E8) already ran locally; their results are in `results/`.
The following are GPU/particle jobs — run on the server (CUDA), then re-run
`python collate.py` to fold the numbers in.

All scripts write `results/<exp_id>/{config.json,metrics.csv,summary.json}` and a
figure in `images/`.

## E7 — 2-D map defect (outer ring), full scale
```
python jko_defect2d.py            # 25 blocks, 12000 particles, 800 epochs
```
Quick sanity (already validated on CPU): `python jko_defect2d.py --quick`.
→ delta_t_normalized_{max,mean}, accumulated_defect_term.

## E9 — dimension scaling on λ=0 targets
```
python jko_neural.py --target iso_quartic  --dims 2,5,10,20,50
python jko_neural.py --target flatvalley   --dims 2,5,10,20,50 --R 2.0
```
Defaults: 25 blocks, 8000 particles, 600 epochs, gamma=0.05, thresh=0.1 (Sinkhorn).
→ blocks_to_thresh_{std,acc}_d<d>, walltime_to_thresh_*, final_W2_*.
Report blocks-to-threshold + wall-clock (targets converge exponentially — no slopes).

## E10 — hyperparameter sensitivity (2-D ring)
```
python jko_sweep.py               # gamma x width x depth = 4x3x3 grid
```
36 cells x 2 methods, 25 blocks, 4000 particles, 300 epochs.
→ results/e10_sensitivity/sensitivity_table.csv (W2 at block 25, std vs acc).

## E11 — Figure 1 reproducibility (2-D outer ring)
```
python jko_densities.py --target outer_ring   # dumps full config.json for the appendix
```
config.json records: prox scheme, exact logdet, extrapolation/momentum, clamps,
seeds, optimizer/schedule, and the geomloss Sinkhorn settings used for the reported W2.

## Notes
- `jko_densities.py` currently selects `cuda` if available else `cpu` (no MPS path);
  on the server CUDA will be picked up automatically.
- E7/E9 subsample to n_sub=2048 for the OT / Sinkhorn steps; POT (`pot`) required.
- `python collate.py` refreshes both the E4 wall-clock table and
  `results/rebuttal_numbers.md` from whatever is in `results/`.
