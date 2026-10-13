# Accelerated Proximal Optimization in Wasserstein Space with a Fixed Reference Measure

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
