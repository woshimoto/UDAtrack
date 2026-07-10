# DriftGuard: State-Safe Referring Multi-Object Tracking

Implementation of **DriftGuard**, including the paper-aligned training setup and
the one-at-a-time sensitivity protocol used in the BMVC rebuttal.

DriftGuard is an end-to-end referring multi-object tracker that protects each
propagated identity in both the read and write steps:

- **Per-track state bank:** every active identity owns a separate semantic state.
- **Track-conditioned evidence admission:** training softly weights all visual
  tokens; inference applies an evidence threshold and a hard per-track budget.
- **Counterfactual supervision:** target evidence is contrasted with box, text
  proposal, and competing-track negatives.
- **Quality-gated writes:** referring confidence, calibrated box quality, and
  association reliability jointly decide whether a state is updated or frozen.

![DriftGuard framework](assets/framework.png)

## News

- The training and inference paths now use the same per-track DriftGuard state.
- `K`, `theta_ev`, and `delta_g` are independent, real inference controls.
- The repository includes an RMOT evaluator and the seven-run sensitivity driver.
- `datasets/` code has been restored so the repository can be imported and trained without relying on ignored local files.
- Training and inference scripts are portable and configurable through environment variables.

## Repository Layout

```text
.
├── main.py                         # distributed training entry
├── inference.py                    # online tracking inference
├── tools/evaluate_rmot.py          # HOTA/AssA/IDSW evaluator
├── tools/run_sensitivity.py        # seven one-at-a-time sensitivity runs
├── configs/                        # runnable train/test shell scripts
├── datasets/                       # RMOT/Refer-KITTI dataset loaders
├── models/
│   ├── transrmot_pro.py            # DriftGuard model and training criterion
│   ├── deformable_transformer_plus.py # evidence router and decoder
│   ├── spatial_temporal_reason.py  # temporal reasoning module
│   └── ops/                        # MultiScaleDeformableAttention CUDA op
├── util/                           # box, checkpoint, plotting, misc utilities
└── TrackEval/                      # HOTA/DetA/AssA evaluation toolkit
```

## Installation

We recommend Python 3.8, CUDA 11.1, and PyTorch 1.9 for compatibility with the Deformable DETR CUDA operator.

```bash
conda create -n driftguard python=3.8 -y
conda activate driftguard

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt

python -m spacy download en_core_web_sm
cd models/ops
sh make.sh
python test.py
```

If your machine is offline, download `roberta-base` in advance and pass it with `--text_encoder_path /path/to/roberta_base --text_encoder_local_files_only`.

## Data Preparation

The loader expects a Refer-KITTI style directory:

```text
Refer-KITTI/
├── images/
├── labels_with_ids/
└── expression/
```

Set the dataset root with `DATA_ROOT` and the split file with `TRAIN_SPLIT`. Example split files are provided in `datasets/data_path/`.

## Training

Refer-KITTI:

```bash
DATA_ROOT=/path/to/Refer-KITTI \
PRETRAIN=/path/to/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
TEXT_ENCODER_PATH=/path/to/roberta_base \
NPROC=8 \
bash configs/dkgtrack_rmot_train_rk.sh
```

Refer-KITTI v2:

```bash
DATA_ROOT=/path/to/refer-kitti-v2 \
TRAIN_SPLIT=/path/to/refer-kitti-v2.train \
PRETRAIN=/path/to/r50_deformable_detr_plus_iterative_bbox_refinement-checkpoint.pth \
TEXT_ENCODER_PATH=/path/to/roberta_base \
bash configs/dkgtrack_rmot_train.sh
```

Both training scripts use 60 epochs, batch size 1 per GPU, and default to eight
GPUs. Important method controls are:

- `--evidence_topk 96`: maximum admitted tokens per active track (`K`).
- `--evidence_score_thresh 0.35`: inference admission threshold (`theta_ev`).
- `--state_update_thresh 0.45`: state-write threshold (`delta_g`).
- `--text_proposal_thresh 0.55`: hard text-proposal threshold (`eta_txt`).
- `--association_margin 0.05`: hard competing-track margin (`m_a`).
- `--quality_beta 2`: IoU exponent used by the quality target.
- `--rectification_strength 0.1`: static/motion language correction strength.
- `--racl_loss_coef 1`: enables RACL in the total loss.
- `--evidence_loss_coef 1`, `--cf_loss_coef 1`, and
  `--quality_loss_coef 1`: the three state-safety objectives.

## Inference

```bash
DATA_ROOT=/path/to/Refer-KITTI \
CHECKPOINT=/path/to/checkpoint.pth \
TEXT_ENCODER_PATH=/path/to/roberta_base \
bash configs/dkgtrack_rmot_test_rk.sh
```

Outputs are written to `OUTPUT_DIR`.

## Evaluation

The paper reports HOTA as the primary metric, together with DetA, AssA, DetRe, DetPr, AssRe, AssPr, and LocA.

```bash
python tools/evaluate_rmot.py \
  --results-root /path/to/eval/results_epoch0059 \
  --output /path/to/eval/metrics.json
```

The evaluator directly reuses the vendored HOTA and CLEAR metric
implementations. It reports HOTA, DetA, AssA, the remaining HOTA submetrics,
and IDSW.

## Sensitivity Experiment

The following command runs exactly seven unique settings: the default, two
alternative values of `K`, two of `theta_ev`, and two of `delta_g`. Every run
uses the same checkpoint and all non-varied settings remain fixed.

```bash
python tools/run_sensitivity.py \
  --data-root /path/to/refer-kitti-v2 \
  --checkpoint /path/to/checkpoint0059.pth \
  --text-encoder-path /path/to/roberta-base \
  --gpus 0 1 2 3 4 5 6 \
  --jobs 7
```

The output directory contains `sensitivity.csv`, `sensitivity.json`, and
`sensitivity_rows.tex`. Verify that the default run reproduces the main-paper
checkpoint before using the remaining rows in a rebuttal.

Run the lightweight routing and metric checks with:

```bash
python -m unittest tests.test_driftguard -v
```

## Implementation Notes

- The per-track state bank is stored on `Instances` and updated by
  `TransRMOT._update_track_states`.
- Track-conditioned routing is implemented in
  `DeformableTransformerDecoderLayer._route_track_evidence`.
- Evidence, counterfactual, and quality supervision are implemented by
  `ClipMatcher.loss_evidence`, `loss_counterfactual`, and `loss_quality`.
- RACL is implemented in `ClipMatcher.loss_contrastive`; query embeddings are
  weighted by IoU reliability and compared against reliable track memory.
- The text encoder path is configurable through `--text_encoder_path` or the `TEXT_ENCODER_PATH` environment variable.

## Acknowledgements

This repository builds on Deformable DETR, MOTR/TransRMOT-style online tracking, DKGTrack-style language decoupling, and TrackEval.

## Citation

If you use this code, please cite the accompanying paper. A BibTeX entry can be added here after publication.
