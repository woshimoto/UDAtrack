#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/path/to/Refer-KITTI}"
TRAIN_SPLIT="${TRAIN_SPLIT:-${PROJECT_ROOT}/datasets/data_path/refer-kitti.train}"
PRETRAIN="${PRETRAIN:-${PROJECT_ROOT}/weights/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth}"
TEXT_ENCODER_PATH="${TEXT_ENCODER_PATH:-roberta-base}"
EXP_DIR="${EXP_DIR:-exps/sgdp_track_refer_kitti}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-23333}"

python3 -m torch.distributed.run --nproc_per_node="${NPROC}" --master_port "${MASTER_PORT}" "${PROJECT_ROOT}/main.py" \
  --meta_arch temp_rmot \
  --use_checkpoint \
  --dataset_file e2e_rmot \
  --epochs 100 \
  --with_box_refine \
  --lr_drop 80 \
  --lr 1e-5 \
  --lr_backbone 1e-5 \
  --pretrained "${PRETRAIN}" \
  --output_dir "${PROJECT_ROOT}/${EXP_DIR}" \
  --save_dir "${PROJECT_ROOT}/${EXP_DIR}" \
  --batch_size 1 \
  --sample_mode random_interval \
  --sample_interval 1 \
  --sampler_steps 60 80 90 \
  --sampler_lengths 4 4 4 4 \
  --update_query_pos \
  --merger_dropout 0.1 \
  --dropout 0 \
  --random_drop 0 \
  --fp_ratio 0.3 \
  --query_interaction_layer QIM \
  --rmot_path "${DATA_ROOT}" \
  --data_txt_path_train "${TRAIN_SPLIT}" \
  --hist_len 4 \
  --refer_loss_coef 2 \
  --racl_loss_coef 1 \
  --racl_beta 2 \
  --racl_temperature 0.07 \
  --racl_num_negatives 50 \
  --cf_loss_coef 1 \
  --unc_loss_coef 0.5 \
  --cf_score_thresh 0.35 \
  --sgdp_topk 300 \
  --sgdp_k_min 64 \
  --text_encoder_path "${TEXT_ENCODER_PATH}" \
  "$@"
