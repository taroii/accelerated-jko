#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."                     # run from the repo root; outputs land in results/ and figures/
SEEDS="${SEEDS:-50}"

python src/jko.py                           # correctness checks; aborts on failure
python src/exp_rate.py       --seeds "$SEEDS"
python src/exp_inexact.py    --seeds "$SEEDS"
python src/exp_geometry.py   --seeds "$SEEDS"
python src/exp_barycenter.py --seeds "$SEEDS"
[ "${WITH_NEURAL:-0}" = "1" ] && python src/exp_neural.py --seeds 10
echo "done"
