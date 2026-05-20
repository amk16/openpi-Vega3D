#!/usr/bin/env bash
#
# Overnight LIBERO finetune. Runs three independent experiments sequentially:
#   1. pi05_libero_lora_wan_precomp          -- pi05 LoRA + VEGA-3D WAN tower
#   2. pi05_libero_lora                      -- plain pi05 LoRA baseline (no VEGA)
#   3. pi05_libero_lora_wan_precomp_semonly  -- VEGA architecture, WAN gated off
# 1 vs 2 = end-to-end value of WAN; 1 vs 3 = clean WAN isolation; 2 vs 3 =
# cost/benefit of the extra P_sem projection. All three stream checkpoints to
# S3 and log a held-out validation loss. A failed run does not stop the next.

# ---- Download dataset + precomputed WAN tower features (must succeed) ----
set -eo pipefail
NVME_DIR=/workspace
HF_HOME="$NVME_DIR/.hf_home"
aws s3 sync s3://behavior-challenge/lerobot/ "$HF_HOME/lerobot/"

python scripts/precompute_tower_features.py pi05_libero_lora_wan_precomp --window 1 --stride 1 --batch_size 128

# Sync into the base tower_features/ dir: the S3 keys already carry the
# physical-intelligence_libero/wan_t2v_16x1536/<camera>/... suffix, which must
# match tower_features_cache_dir in the pi05_libero_lora_wan_precomp config.
aws s3 sync s3://behavior-challenge/tower_features/physical-intelligence_libero/wan_t2v_16x1536_w1s1_blk20 /workspace/openpi-Vega3D/tower_features/physical-intelligence_libero/wan_t2v_16x1536_w1s1_blk20

# Past this point a failed training run must not abort the script.
set +e
source /venv/main/bin/activate

# ---- Run 1: WAN-tower variant (precomputed features) ----
echo "[run_libero_finetune] $(date) starting WAN run (pi05_libero_lora_wan_precomp)"
HF_HOME=$HF_HOME XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
    pi05_libero_lora_wan_precomp \
    --exp-name=wan_precomp_v1_w1s1_blk20 \
    --overwrite
echo "[run_libero_finetune] $(date) WAN run exited with code $?"

# ---- Run 2: plain pi05 LoRA baseline (no VEGA) ----
echo "[run_libero_finetune] $(date) starting baseline run (pi05_libero_lora)"
HF_HOME=$HF_HOME XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
    pi05_libero_lora \
    --exp-name=lora_baseline_v1_w1s1_blk20 \
    --overwrite
echo "[run_libero_finetune] $(date) baseline run exited with code $?"

# ---- Run 3: WAN ablation control (VEGA architecture, WAN gated off) ----
echo "[run_libero_finetune] $(date) starting WAN-control run (pi05_libero_lora_wan_precomp_semonly)"
HF_HOME=$HF_HOME XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py \
    pi05_libero_lora_wan_precomp_semonly \
    --exp-name=wan_precomp_semonly_v1_w1s1_blk20 \
    --overwrite
echo "[run_libero_finetune] $(date) WAN-control run exited with code $?"
