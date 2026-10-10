#!/usr/bin/env bash
# Unit tests, then E2 -> E1 -> E3 -> E4 (the order of the spec; E2 fixes the channel ratio used by E3/E4).
#   ./scripts/run_all.sh                 full spec runs into results/
#   QUICK=1 ./scripts/run_all.sh         smoke test (small grids, few steps) into results_smoke/, ~1 min
#   JOBS=12 ./scripts/run_all.sh         run the E3 targets / E4 configurations in parallel processes
#   EXPS="e2 e3" ./scripts/run_all.sh    only the named experiments
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
JOBS="${JOBS:-1}"
EXPS="${EXPS:-e2 e1 e3 e4}"
if [ "${QUICK:-0}" = "1" ]; then
  export AJKO_RESULTS=results_smoke
  A_e2="--n 32 --n-big 48 --iters 400 --ref-iters 2000"
  A_e1="--T 30 --sweep-steps 20"
  A_e3="--n 32 --T 12 --ref-steps 40 --jobs $JOBS"
  A_e4="--n 32 --T 5 --cap 200 --jobs $JOBS"
else
  A_e2=""; A_e1=""; A_e3="--jobs $JOBS"; A_e4="--jobs $JOBS"
fi
OUT="${AJKO_RESULTS:-results}"
mkdir -p "$OUT"
$PY src/test_units.py | tee "$OUT/unit_tests.txt"      # aborts here if any test fails
for e in $EXPS; do
  case "$e" in
    e1) $PY src/e1_gaussian_rates.py $A_e1 ;;
    e2) $PY src/e2_inner_solver.py   $A_e2 ;;
    e3) $PY src/e3_end_to_end.py     $A_e3 ;;
    e4) $PY src/e4_mixture.py        $A_e4 ;;
    *)  echo "unknown experiment: $e" >&2; exit 1 ;;
  esac
done
echo "done -> $OUT/"
