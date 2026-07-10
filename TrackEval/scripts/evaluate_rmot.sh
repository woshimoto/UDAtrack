#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
RESULTS_ROOT="${RESULTS_ROOT:-${1:-${PROJECT_ROOT}/exps/driftguard_refer_kitti_v2/eval/results_epoch0059}}"
OUTPUT_JSON="${OUTPUT_JSON:-${RESULTS_ROOT}/metrics.json}"

python3 "${PROJECT_ROOT}/tools/evaluate_rmot.py" \
  --results-root "${RESULTS_ROOT}" \
  --output "${OUTPUT_JSON}"
