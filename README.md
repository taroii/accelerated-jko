# Accelerated Proximal Optimization in Wasserstein Space with a Fixed Reference Measure

Code for the experiments of Section 6 (E1–E4). The specification the code implements is
`handoff/experiments/EXPERIMENTS.md`; the write-up is `handoff/paper/`.

Everything is CPU, float64, deterministic (no random numbers enter any experiment), and needs
only numpy, scipy and matplotlib.

## Install

```
pip install -r requirements.txt          # numpy scipy matplotlib
```

## Reproduce

```
./scripts/run_all.sh                 # unit tests, then E2 -> E1 -> E3 -> E4, full spec, into results/
QUICK=1 ./scripts/run_all.sh         # smoke test (small grids, few steps) into results_smoke/, about a minute
JOBS=12 ./scripts/run_all.sh         # E3 targets / E4 configurations in parallel processes
EXPS="e2 e3" ./scripts/run_all.sh    # only the named experiments
PYTHON=/path/to/python ./scripts/run_all.sh
```

The unit tests run first and the script aborts if any fails. E2 must run before E3/E4: it
writes the channel ratio `results/e2/r_best.json` that they read (they fall back to 10 and say
so if the file is missing). Each script also runs on its own with no arguments
(`python src/e3_end_to_end.py`); the defaults are the spec.

Rough single-core cost: unit tests 10 s, E1 1 min, E2 4 min, E3 minutes to an hour, E4 the
slowest (12 runs x 80 outer steps x up to 2000 inner iterations) — use `JOBS`.

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

## Status of `results/`

`results/e1` and `results/e2` are full spec runs. E3 and E4 have only been smoke-tested
(`QUICK=1`); their full runs are still to be done.

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
   weight instead of a single 12-curve panel.
9. **`--jobs`** (off by default) runs independent configurations in separate processes.

Released under the MIT License (see `LICENSE`).
