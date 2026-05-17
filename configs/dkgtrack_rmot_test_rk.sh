#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/path/to/Refer-KITTI}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/exps/sgdp_track_refer_kitti/checkpoint.pth}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-roberta-base}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/exps/sgdp_track_refer_kitti/eval}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES

python3 "${PROJECT_ROOT}/inference.py" \
  --meta_arch temp_rmot \
  --dataset_file e2e_rmot \
  --with_box_refine \
  --batch_size 1 \
  --sample_mode random_interval \
  --sample_interval 1 \
  --sampler_steps 50 90 150 \
  --sampler_lengths 2 3 4 5 \
  --update_query_pos \
  --merger_dropout 0 \
  --dropout 0 \
  --random_drop 0.1 \
  --fp_ratio 0.3 \
  --query_interaction_layer QIM \
  --extra_track_attn \
  --hist_len 8 \
  --rmot_path "${DATA_ROOT}" \
  --resume "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --sgdp_topk 300 \
  --sgdp_k_min 64 \
  --text_encoder_path "${TEXT_ENCODER_PATH}" \
  "$@"
