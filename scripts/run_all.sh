#!/usr/bin/env bash
# Unit tests, then E2 -> E1 -> E3 -> E4 (the order of the spec; E2 fixes the channel ratio used by E3/E4).
# Each experiment runs the fixed instance of the spec and then an ensemble over random instances
# (seed = instance; 100 / 50 / 20 / 20 seeds for E1 / E2 / E3 / E4).
#   ./scripts/run_all.sh                 full runs into results/, on half the CPU cores at low priority
#   QUICK=1 ./scripts/run_all.sh         smoke test (small grids, few steps, 3 seeds) into results_smoke/
#   JOBS=4 ./scripts/run_all.sh          number of parallel processes (default: half the cores)
#   EXPS="e2 e3" ./scripts/run_all.sh    only the named experiments
#   RESUME=1 ./scripts/run_all.sh        continue interrupted ensembles from their per-seed files
#   SEEDS=0 ./scripts/run_all.sh         fixed instances only (or any seed count for all four)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"
NCPU="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-2}")"
JOBS="${JOBS:-$(( NCPU / 2 > 0 ? NCPU / 2 : 1 ))}"     # half the cores by default; workers run at low priority
COMMON="--jobs $JOBS"
[ "${RESUME:-0}" = "1" ] && COMMON="$COMMON --resume"
EXPS="${EXPS:-e2 e1 e3 e4}"
if [ "${QUICK:-0}" = "1" ]; then
  export AJKO_RESULTS=results_smoke
  SEEDS="${SEEDS:-3}"
  A_e2="--n 32 --n-big 48 --iters 400 --ref-iters 2000"
  A_e1="--T 30 --sweep-steps 20 --sweep-seeds 3"
  A_e3="--n 32 --T 12 --ref-steps 40"
  A_e4="--n 32 --T 5 --cap 200"
else
  A_e2=""; A_e1=""; A_e3=""; A_e4=""
fi
[ -n "${SEEDS:-}" ] && COMMON="$COMMON --seeds $SEEDS"
OUT="${AJKO_RESULTS:-results}"
mkdir -p "$OUT"
export PYTHONUNBUFFERED=1
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $* ==="; }
stamp "unit tests"
$PY src/test_units.py > "$OUT/unit_tests.txt" || { cat "$OUT/unit_tests.txt"; exit 1; }   # aborts if any test fails
tail -1 "$OUT/unit_tests.txt"
for e in $EXPS; do
  stamp "$e  (jobs: $JOBS, experiments queued: $EXPS)"
  case "$e" in
    e1) $PY src/e1_gaussian_rates.py $A_e1 $COMMON ;;
    e2) $PY src/e2_inner_solver.py   $A_e2 $COMMON ;;
    e3) $PY src/e3_end_to_end.py     $A_e3 $COMMON ;;
    e4) $PY src/e4_mixture.py        $A_e4 $COMMON ;;
    *)  echo "unknown experiment: $e" >&2; exit 1 ;;
  esac
done
stamp "all done -> $OUT/"
