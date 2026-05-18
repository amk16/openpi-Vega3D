#!/usr/bin/env bash

# Download dataset
NVME_DIR=/workspace
HF_HOME="$NVME_DIR/.hf_home"
aws s3 sync s3://behavior-challenge/lerobot/ "$HF_HOME/lerobot/"
aws s3 sync s3://behavior-challenge/tower_features/ /workspace/openpi-Vega3D/tower_features/physical-intelligence_libero/wan_t2v_16x1536

# Activate venv
source /venv/main/bin/activate

# Kick off training run
HF_HOME=$HF_HOME XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
    pi05_libero_lora \
    --exp-name=pi05_libero_lora_base \
    --overwrite
