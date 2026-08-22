# Accelerated Proximal Optimization in Wasserstein and Sobolev Spaces

Reference implementation and experiments for the accelerated JKO / Sobolev-gradient-ascent
framework. The suite validates the theory with **exact proximal oracles in arbitrary
dimension** — no neural network and no sampling-based OT solver on the critical path — so
every quantitative result is deterministic given its inputs and runs on a CPU.

## Install

```
pip install -r requirements.txt          # numpy scipy matplotlib
```

`torch` and `geomloss` are needed only for the optional neural demonstration
(`exp_neural.py`); the four quantitative experiments do not use them.

## Reproduce

```
./scripts/run_all.sh                 # 50 seeds, CPU, well under an hour
SEEDS=2 ./scripts/run_all.sh         # smoke test, < 2 minutes
WITH_NEURAL=1 ./scripts/run_all.sh   # also runs the optional GPU demo
```

`run_all.sh` first runs `python src/jko.py`, the correctness suite (Section below); it aborts
if any check fails. Each experiment writes `results/<exp>/{config.json,per_seed.csv,summary.json}`
and `figures/<exp>.pdf` at the repo root. Figures and summaries are regenerable from
`per_seed.csv` alone:

```
python src/exp_rate.py --from-csv results/rate/per_seed.csv
```

## Layout

```
src/       jko.py + one file per experiment
scripts/   run_all.sh
results/   generated per-experiment output (config.json, per_seed.csv, summary.json)
figures/   generated PDF figures
```

| file | contents |
|---|---|
| `src/jko.py` | exact oracles (`prox_radial`, `prox_interaction`, `prox_gaussian`, `prox_quantile`), Bures/Gaussian geometry (`bw_map`, `bw_distance`, `map_defect`), the standard/accelerated schemes (`run`), paired statistics (`paired_summary`, `bootstrap_exponent`), figure style, and the correctness checks (`python src/jko.py`) |
| `src/exp_rate.py` | **Experiment 1** — rate under vanishing curvature: blocks-to-threshold vs a degeneracy parameter `R`, exact in 1-D (quantile) and any-`d` (Gaussian); non-diffusive potential/interaction arms for the unconditional `d ≥ 2` corollary; a non-log-concave control |
| `src/exp_inexact.py` | **Experiment 2** — inexact proximal steps: controlled per-step error `e_t ∝ t^{-p/2}`, the `Θ(Tε)` accumulation / transition at `p = 3`, and the theorem's right-hand-side bound |
| `src/exp_geometry.py` | **Experiment 3** — where the hypotheses bind: exact Gaussian map defect for `d ≥ 2`, the measured bi-Lipschitz condition `λ_min(sym Z_{t+1})`, and the potential-vs-entropy case split |
| `src/exp_barycenter.py` | **Experiment 4** — barycenter dual: ASGA vs SGA slope distributions over random instances, certified vs empirical step size |
| `src/exp_neural.py` | **Experiment 5** (optional) — a qualitative neural particle-transport demonstration; the only file importing `torch` |

## Statistics

Randomness enters only through problem instances and initial conditions; standard and
accelerated **share the instance within a seed**, so every comparison is paired. Curves are
reported as median across seeds with a shaded IQR band; scaling exponents and method
comparisons come with bootstrap CIs and Wilcoxon signed-rank tests over the 50 seeds. Seeds
`0..49` index the initial condition or problem instance (the Gaussian and quantile arms are
deterministic given their inputs). Raw per-seed output is committed so every summary is
recomputable from the CSV without re-running.

## Correctness checks

`python src/jko.py` verifies, among others: `prox_radial` against a `scipy` reference; the
codiagonal Gaussian prox against the general L-BFGS path; that the map defect is `0` for
commuting (codiagonal) maps and strictly positive after a rotation; and that the accelerated
step with `α = 1` reduces to a standard JKO step.

Released under the MIT License (see `LICENSE`).
