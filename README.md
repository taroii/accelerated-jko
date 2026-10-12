# Accelerated Proximal Optimization in Wasserstein Space with a Fixed Reference Measure

Code for the experiments of Section 6 (E1–E4). The specification the code implements is
`handoff/experiments/EXPERIMENTS.md`; the write-up is `handoff/paper/`.

Everything is CPU, float64, reproducible from fixed seeds, and needs only numpy, scipy and
matplotlib.

## Install

```
pip install -r requirements.txt          # numpy scipy matplotlib
```

## Reproduce

```
./scripts/run_all.sh                 # unit tests, then E2 -> E1 -> E3 -> E4 into results/, on half the cores at low priority
QUICK=1 ./scripts/run_all.sh         # smoke test (small grids, few steps, 3 seeds) into results_smoke/, about a minute
JOBS=4 ./scripts/run_all.sh          # number of parallel processes (default: half the cores)
EXPS="e2 e3" ./scripts/run_all.sh    # only the named experiments
RESUME=1 ./scripts/run_all.sh        # continue interrupted ensembles from their per-seed files
SEEDS=0 ./scripts/run_all.sh         # fixed spec instances only (or any seed count for all four)
PYTHON=/path/to/python ./scripts/run_all.sh
```

The unit tests run first and the script aborts if any fails. E2 must run before E3/E4: it
writes the channel ratio `results/e2/r_best.json` that they read (they fall back to 10 and say
so if the file is missing). Each script also runs on its own with no arguments
(`python src/e3_end_to_end.py`); the defaults are the spec, single process.

Each experiment has two parts, both written to the same `results.md`:

1. the **canonical** fixed instance of the spec (`<exp>.npz`, `fig_<exp>.*`);
2. an **ensemble** over random problem instances (`<exp>_ensemble.npz`, `fig_<exp>_ens.*`).

Rough cost in CPU-hours (divide by the core count): E1 about 1.5, E2 about 0.5, E3 about 1,
E4 about 16 (4 ms per inner iteration, and nearly every step runs to the 2000-iteration cap).

## Statistics

The algorithms are deterministic; the randomness is over **problem instances**. A seed fixes
the reference, the target and (in E2) the field Y, and every method runs on that same instance,
so comparisons are paired. Seed `s` of experiment `k` uses `numpy.random.default_rng([k, s])`.

| exp | seeds | what a seed draws |
|---|---|---|
| E1 | 100 (+50 per lambda in the sweep) | `Zc = R(th) diag(e^a1, e^a2) R(th)^T`, `th ~ U(0, pi)`, `a_i ~ U(0, log 10)`; `Q = R(th') diag(1/lam_i) R(th')^T`, `lam_i` log-uniform on [0.02, 0.2] (weak) or [0.5, 5] (strong) |
| E2 | 50 | Gaussian target, centre `~ U([0.35, 0.65]^2)`, variances log-uniform on [0.01, 0.06]; `Y = x + 0.08 (grad phi + curl chi)`, random cosine-series potentials with coefficients `N(0,1)/|k|^3` (`|k|_inf <= 6`, unit sup norm) |
| E3 | 20 | Gaussian target as in E2 (uniform reference, `X_0 = Id`) |
| E4 | 20 | `w ~ U(0.6, 0.95)`, modes at `(0.5 -+ delta, 0.5)` with `delta ~ U(0.15, 0.25)`, offset of the asymmetric reference `~ N(0, 0.02^2)`; references A (uniform) and C (offset Gaussian) |

Reported quantities are scale-free: in E1 the bound utilisation
`u_t = gap_t gamma (t+1)^2 / (2 W_{2,nu}^2(x_0, rho*))` (Theorem 1: `u_t <= 1` on every
instance); in E2–E4 the gap divided by the initial gap, iterations to a tolerance, and paired
log-ratios accelerated / non-accelerated at fixed `t`. Curves are the median with a 2.5–97.5
percentile band; tables give the mean with a 95% bootstrap CI of the mean (2000 resamples), and
for iteration counts the median with a bootstrap CI of the median, with runs that never reach
the tolerance right-censored and the fraction reached stated. E4 reports the mode recovery rate
(left-half-plane mass within 0.05 of `w`) with a Wilson 95% interval.

Choices the design left open: the E2/E3 ensembles use the Gaussian target only (the quartic
stays in the canonical runs); `r_best` still comes from the canonical E2 instance (the ensemble's
own best ratio is reported next to it); the E2 n=128 run and the E4 evolution figure are
canonical only. Per-seed raw output is kept in `results/<exp>/seeds*/` (git-ignored) for `RESUME`.

## Layout

```
src/        shared modules, unit tests, one script per experiment
scripts/    run_all.sh
results/    e1/ e2/ e3/ e4/ — each with <exp>.npz (raw curves), fig_*.pdf/png, results.md; unit_tests.txt
handoff/    the specification, the paper source, and the earlier prototypes
```

| file | contents |
|---|---|
| `src/grid.py` | cosine-series collocation on `[0,1]^2`: `K1` (Hessian), `K2` (gradient), `B` (`K1 = B K2`), exact adjoints; targets `Gauss`, `Quartic`, `Mix2` with their pointwise transport solves and conjugates |
| `src/pdhg.py` | Algorithm 2 (two-channel spectrally preconditioned PDHG, plus the scalar-step baseline), the duality certificate, and Algorithm 1 on the grid (`run_outer`) |
| `src/gaussian.py` | exact 2x2 Gaussian machinery: OT maps, restricted proximal step over SPD maps, JKO step, the four schemes of E1 |
| `src/common.py` | output paths, plot style, reference curves and the bound of Theorem 1 |
| `src/test_units.py` | transforms, adjoints, multipliers, proxes, conjugates, weak duality, Gaussian solvers |
| `src/e1_gaussian_rates.py` | **E1** — rates with exact steps; lambda sweep |
| `src/e2_inner_solver.py` | **E2** — inner solver: scalar vs preconditioned, channel ratio, balanced steps, n-scaling, certificate |
| `src/e3_end_to_end.py` | **E3** — outer scheme with the inner solver, certified schedule vs fixed budget |
| `src/e4_mixture.py` | **E4** — non-log-concave mixture target, three reference measures |
| `src/paper_figures.py` | figures and tables for the paper, built from the saved results into `results/figures/` (run after the experiments) |

## Status of `results/`

`results/e1`, `e2` and `e4` hold full runs (canonical instances and ensembles). `results/e3` must be
rerun after the change of protocol in deviation 11, and `results/e1` once more for the flat
targets (`e1_flat.npz`): `RESUME=1 EXPS="e1 e3" ./scripts/run_all.sh`, then
`python src/paper_figures.py`. E4 is kept as a record but is not used in the paper: neither
scheme recovers the mode weights (0/20 instances).

## Deviations from the spec

1. **Certificate (Remark 4).** The certificate as specified,
   `J - (-h1*(Lam) - h2*(mu) - R ||B^T Lam + mu||_1)`, is a valid bound but does not vanish:
   at a saddle point only `K2^T (B^T Lam + mu) = 0` holds, and `r = B^T Lam + mu` keeps a
   component orthogonal to gradient fields (it stays at about 3.3 and 1.6 on the two E2 targets).
   Because `<K2 psi, r> = <K2 psi, P r>` with `P` the orthogonal projection onto `range(K2)`,
   `R ||P r||_1` is an equally valid bound, costs one more DCT pair, and does go to zero. All
   stopping rules use the projected certificate; E2 records both.
2. **Restricted Gaussian prox.** BFGS and the Nelder–Mead polish are as specified, followed by
   a Newton iteration on the 3x3 KKT system, which is what brings the residual to ~1e-15.
3. **Normalisation.** `V` includes `log Z`, so `q = e^{-V}` is a density on the box and
   `G = KL(x | q)`. Gaps are unaffected.
4. **Feasibility.** An iterate counts as admissible only if, in addition to `I + Hess psi > 0`,
   the map stays in the box at every cell (`V = +inf` outside).
5. **Weights.** Cell weights are stored as `nu(x)/mean(nu)` (1 for the uniform reference);
   reported objectives, gaps, certificates and tolerances are multiplied by `h^2`.
6. **1-D transport solve.** Bracketed Newton run to convergence (bisection whenever a step
   leaves the bracket) rather than exactly 10 Newton steps.
7. **`r_best`** is the argmin of iterations-to-1e-6 summed over the two E2 targets.
8. **E4.** The mixture conjugate `h2*` is a best-of-two-starts local maximisation, so the
   certificate is only a heuristic stopping rule there. The gap figure has one panel per mixture
   weight instead of a single 12-curve panel. The certificate never reaches the tolerance there, so an inner
   solve also ends once its coefficients move by less than 1e-10 (relative) over 25 iterations
   (reported as 'steps stalled'); this only removes iterations that no longer change the iterate.
9. **`--jobs`** runs independent configurations and seeds in separate processes (default 1 in the
   scripts, half the cores in `run_all.sh`); wall-clock columns are then noisier.
10. **Ensembles** over random instances are an addition to the spec (see Statistics).
11. **E3 step size.** E3 runs at `gamma = 5e-4` for 300 outer steps instead of `gamma = 0.05` for
    60. At 0.05 the targets have `gamma * lambda >= 1`, every scheme converges in under 30 steps
    and acceleration cannot help (that run is in the git history). The reference run for `G*`
    keeps `gamma = 0.05`. Certified solves use the stall rule of deviation 8, since the schedule
    falls below what the certificate resolves. Per-seed files are in `seeds_gamma0.0005/`.
12. **E1 flat targets.** E1 also runs `Q = I/lam` for `lam = 1e-3, 1e-2, 1e-1` over 1000 steps
    (`e1_flat.npz`, `fig_e1_flat.*`), the regime where the two rates separate.

Released under the MIT License (see `LICENSE`).
