#!/usr/bin/env bash
#
# LIBERO finetune with Cosmos-Policy-LIBERO tower features.

# ---- Download dataset + precomputed tower features (must succeed) ----
set -eo pipefail
NVME_DIR=/workspace
HF_HOME="$NVME_DIR/.hf_home"
aws s3 sync s3://behavior-challenge/lerobot/ "$HF_HOME/lerobot/"

aws s3 sync \
    s3://behavior-challenge/tower_features/physical-intelligence_libero/cosmos_policy_libero_16x2048_w1s1_blk20_t5cond \
    /workspace/openpi-Vega3D/tower_features/physical-intelligence_libero/cosmos_policy_libero_16x2048_w1s1_blk20_t5cond

# Past this point a failed training run must not abort the script.
set +e
source /venv/main/bin/activate

echo "[run_libero_finetune] $(date) starting Cosmos-Policy-LIBERO FFT run"
OPENBLAS_NUM_THREADS=1 HF_HOME=$HF_HOME XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
    pi05_libero_fft_cosmos_policy_libero_precomp \
    --exp-name=fft_cosmos_policy_libero_precomp \
    --overwrite
echo "[run_libero_finetune] $(date) Cosmos-Policy-LIBERO run exited with code $?"
