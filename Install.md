# Installation Guide

Use Python 3.8 and CUDA 11.1 for the closest match to the original environment.

```bash
conda create -n sgdp-track python=3.8 -y
conda activate sgdp-track

pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m spacy download en_core_web_sm

cd models/ops
sh make.sh
python test.py
```

For offline servers, download `roberta-base` first and run scripts with:

```bash
TEXT_ENCODER_PATH=/path/to/roberta_base bash configs/dkgtrack_rmot_train_rk.sh
```
