#!/usr/bin/env bash
# Unit tests, then E2 -> E1 -> E3 -> E4 (the order of the spec; E2 fixes the channel ratio used by E3/E4).
# Each experiment runs the fixed instance of the spec and then an ensemble over random instances
# (seed = instance; 100 / 50 / 20 / 20 seeds for E1 / E2 / E3 / E4).
#   ./scripts/run_all.sh                 full runs into results/, on half the CPU cores at low priority
#   QUICK=1 ./scripts/run_all.sh         smoke test (small grids, few steps, 3 seeds) into results_smoke/
#   JOBS=4 ./scripts/run_all.sh          number of parallel processes (default: half the cores)
#   EXPS="e2 e3" ./scripts/run_all.sh    only the named experiments (e5, the step-size sweep, runs only when named)
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
  A_e1="--T 30 --sweep-steps 20 --sweep-seeds 3 --flat-steps 30"
  A_e3="--n 32 --T 12 --ref-steps 40"
  A_e4="--n 32 --T 5 --cap 200"
  A_e5="--n 32 --iters 300 --ref-iters 1500 --T-max 60 --budget 3000 --stall-steps 20 --ref-steps 40"
else
  A_e2=""; A_e1=""; A_e3=""; A_e4=""; A_e5=""
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
    e1) f=e1_gaussian_rates.py; A="$A_e1" ;;
    e2) f=e2_inner_solver.py;   A="$A_e2" ;;
    e3) f=e3_end_to_end.py;     A="$A_e3" ;;
    e4) f=e4_mixture.py;        A="$A_e4" ;;
    e5) f=e5_stepsize.py;       A="$A_e5" ;;
    *)  echo "unknown experiment: $e" >&2; exit 1 ;;
  esac
  rc=0
  $PY "src/$f" $A $COMMON || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!! [$(date +%H:%M:%S)] $e FAILED with exit code $rc — stopping. Rerun with: RESUME=1 EXPS=\"...\" $0" >&2
    exit "$rc"
  fi
  echo "--- [$(date +%H:%M:%S)] $e finished"
done
stamp "all done -> $OUT/"
