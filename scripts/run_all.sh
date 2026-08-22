#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."                     # run from the repo root; results/ committed, figures land in paper/
SEEDS="${SEEDS:-50}"
FLAG="--seeds $SEEDS"
[ "${QUICK:-0}" = "1" ] && FLAG="--quick"   # QUICK=1 -> 2-seed smoke, < 2 min

python src/jko.py                           # correctness checks; aborts on failure
python src/exp_rate.py       $FLAG
python src/exp_inexact.py    $FLAG
python src/exp_geometry.py   $FLAG
python src/exp_barycenter.py $FLAG
[ "${WITH_NEURAL:-0}" = "1" ] && python src/exp_neural.py ${QUICK:+--quick}
echo "done"
