#!/usr/bin/env bash
#
# Overnight Cosmos-base LIBERO training chain. Three stages:
#
#   1. precompute      — If the S3 cache is already populated for all 1693
#                        episodes × 2 cameras, this stage is a no-op (no
#                        Python launched, no GPU touched). Otherwise:
#                        ensure cosmos_prompt_embeddings.pt exists (regenerate
#                        from T5-11B if missing), then encode all LIBERO
#                        frames through Cosmos-Predict2.5-2B tower at
#                        single-frame (w=1, s=1), upload to S3, delete
#                        local as we go (disk too tight otherwise).
#                        Idempotent: re-running resumes from S3.
#   2. sync_down       — `aws s3 sync` the cache back from S3 to local disk
#                        so training can read it from local safetensors.
#                        Note: S3 prefix has `_t5cond` suffix because
#                        --prompt_cache was passed; local cache dir does NOT.
#   3. train           — pi05 LIBERO + deep LoRA rank 32 + EMA 0.999 + cosmos_base
#                        tower. No KI, no FAST aux. 30k steps, batch 32,
#                        single-frame precomputed features.
#
# Observed wall clock on H200 (192-core box): stage 1 ~5 hr, stage 2 ~1.5 hr,
# stage 3 ~6-10 hr (extrapolating from prior 30k-step Wan runs). Mileage
# varies on other hardware.
#
# Prerequisites on the target machine (must be set up before invoking this):
#   - Cosmos-Predict2.5-2B downloaded to /workspace/openpi-Vega3D/ckpts/Cosmos-Predict2.5-2B/
#     with `model_ema_bf16.pt` at top level and
#     `vae/{config.json, diffusion_pytorch_model.safetensors}`. See
#     scripts/precompute_tower_features.py:ensure_cosmos_base_checkpoint for the recipe.
#     The VAE lives on the `diffusers/base/post-trained` revision of the HF repo.
#   - HF auth: nvidia/Cosmos-Predict2.5-2B accepted + google-t5/t5-11b readable.
#   - AWS creds: `aws s3 ls s3://behavior-challenge/` succeeds (RW).
#   - ≥600 GiB free on /workspace.
#
# Designed to run inside tmux:
#   tmux new -s cosmos -d 'bash scripts/run_cosmos_base_overnight.sh 2>&1 | tee overnight_cosmos.log'

set -eo pipefail
cd /workspace/openpi-Vega3D
source /venv/main/bin/activate

CONFIG=pi05_libero_deeplora_cosmos_base_precomp_w1s1
EXP=cosmos_base_v1_w1s1
S3_PREFIX=s3://behavior-challenge/tower_features/physical-intelligence_libero/cosmos_base_16x2048_w1s1_blk20_t5cond
LOCAL_CACHE=/workspace/openpi-Vega3D/tower_features/physical-intelligence_libero/cosmos_base_16x2048_w1s1_blk20
PROMPT_CACHE=src/openpi_vega3d/towers/cosmos_prompt_embeddings.pt
EXPECTED_EPISODES=1693   # LIBERO episode count; precompute writes 1 safetensors per (episode, camera)

echo "[$(date)] === Pre-flight ==="
df -h /workspace
du -sh /workspace/openpi-Vega3D/{checkpoints,ckpts,tower_features} 2>&1 || true
aws s3 ls s3://behavior-challenge/ > /dev/null
echo "  S3 access OK"

echo ""
echo "[$(date)] === Stage 1/3: precompute Cosmos tower features ==="
# Short-circuit: if S3 already has every episode for both cameras, skip the
# precompute entirely -- no need to load T5/Cosmos, the dataset, or even Python.
# precompute_tower_features.py itself resumes from S3, but that still costs
# ~30-60s of tower/dataset init even when there's no work to do.
S3_BASE_COUNT=$(aws s3 ls "$S3_PREFIX/base_0_rgb/" 2>/dev/null | grep -c "\.safetensors$" || true)
S3_WRIST_COUNT=$(aws s3 ls "$S3_PREFIX/left_wrist_0_rgb/" 2>/dev/null | grep -c "\.safetensors$" || true)
if [ "$S3_BASE_COUNT" = "$EXPECTED_EPISODES" ] && [ "$S3_WRIST_COUNT" = "$EXPECTED_EPISODES" ]; then
    echo "  S3 already has $EXPECTED_EPISODES episodes for both cameras at $S3_PREFIX"
    echo "  Skipping precompute. Will sync down in stage 2."
else
    echo "  S3 has base=$S3_BASE_COUNT, wrist=$S3_WRIST_COUNT (target $EXPECTED_EPISODES each). Running precompute."
    # Stage 0a: ensure Cosmos T5 prompt embeddings exist (precompute needs them).
    if [ -f "$PROMPT_CACHE" ]; then
        echo "  $PROMPT_CACHE already present, skipping T5 export."
    else
        echo "  $PROMPT_CACHE missing -- generating from T5-11B (~5-10 min after model download)."
        python3 scripts/export_cosmos_prompt_embeddings.py \
            --repo_id physical-intelligence/libero \
            --out_path "$PROMPT_CACHE"
    fi
    # Window=1 stride=1 is set in the config; --prompt_cache adds T5 conditioning.
    # Local files deleted as we go (disk too tight to keep all ~534 GiB).
    python3 scripts/precompute_tower_features.py "$CONFIG" \
        --prompt_cache "$PROMPT_CACHE"
fi
echo "[$(date)] Stage 1 OK"

echo ""
echo "[$(date)] === Stage 2/3: sync cache from S3 back to local ==="
mkdir -p "$LOCAL_CACHE"
aws s3 sync "$S3_PREFIX" "$LOCAL_CACHE"
echo "[$(date)] Stage 2 OK"
du -sh "$LOCAL_CACHE"

echo ""
echo "[$(date)] === Stage 3/3: train ==="
# XLA mem fraction set high since the tower is NOT in GPU
# (vega3d_skip_tower_construction=True); only PaliGemma + action expert
# + LoRA adapters live there. TOKENIZERS_PARALLELISM=false is set at the
# top of scripts/train.py to suppress the rayon thread burst that caused
# the previous stage-3 crash on this hardware.
HF_HOME=/workspace/.hf_home \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
python3 scripts/train.py "$CONFIG" \
    --exp-name "$EXP" \
    --overwrite

echo "[$(date)] Stage 3 OK"
echo "[$(date)] === Overnight chain complete ==="
