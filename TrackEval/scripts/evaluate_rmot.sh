#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SEQMAP_FILE="${SEQMAP_FILE:-${PROJECT_ROOT}/datasets/data_path/seqmap.txt}"
GT_FOLDER="${GT_FOLDER:-/path/to/Refer-KITTI/training/image_02}"
TRACKERS_FOLDER="${TRACKERS_FOLDER:-${PROJECT_ROOT}/exps/sgdp_track_refer_kitti/eval}"
TRACKERS_TO_EVAL="${TRACKERS_TO_EVAL:-${TRACKERS_FOLDER}}"
NUM_PARALLEL_CORES="${NUM_PARALLEL_CORES:-2}"

python3 "${PROJECT_ROOT}/TrackEval/scripts/run_mot_challenge.py" \
  --METRICS HOTA \
  --SEQMAP_FILE "${SEQMAP_FILE}" \
  --SKIP_SPLIT_FOL True \
  --GT_FOLDER "${GT_FOLDER}" \
  --TRACKERS_FOLDER "${TRACKERS_FOLDER}" \
  --GT_LOC_FORMAT "{gt_folder}{video_id}/{expression_id}/gt.txt" \
  --TRACKERS_TO_EVAL "${TRACKERS_TO_EVAL}" \
  --USE_PARALLEL True \
  --NUM_PARALLEL_CORES "${NUM_PARALLEL_CORES}" \
  --PLOT_CURVES False
