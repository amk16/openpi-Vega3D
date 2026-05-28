#!/usr/bin/env bash
#
# Sequentially evaluate several (config, checkpoint) pairs on a set of LIBERO
# suites by invoking scripts/run_libero_eval.sh for each combination. Each
# invocation starts its own inference server and tears it down before the next
# one starts, so there is never more than one pi05 server running at a time.
#
# Edit the RUNS and SUITES arrays below for your sweep. A failure in any single
# (run, suite) does not stop the rest of the sweep.
#
# Usage:
#   bash scripts/sequence_libero_evals.sh [extra args forwarded to libero main]
#
# Example extra args:
#   --args.num-trials-per-task 10

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="$SCRIPT_DIR/run_libero_eval.sh"

# Each entry: "label|config|checkpoint_dir"
#   label          — short name used for the per-run video / results directory
#   config         — training config name (passed to --policy.config)
#   checkpoint_dir — s3://, gs://, or local path. s3:// paths are synced into the
#                    local checkpoint cache by run_libero_eval.sh before serving.

# NOTE: These configs now use precomp names + --load-live-tower flag for inference.
# "wan_precomp_original|pi05_libero_lora_wan_precomp_last_blk|s3://behavior-challenge/openpi_checkpoints/pi05_libero_lora_wan_precomp/wan_precomp_v1/8000"
# "wan_precomp_w1s1|pi05_libero_lora_wan_precomp|s3://behavior-challenge/openpi_checkpoints/pi05_libero_lora_wan_precomp/wan_precomp_v1_w1s1_blk20/8000"
# "lora_baseline|pi05_libero_lora|s3://behavior-challenge/openpi_checkpoints/pi05_libero_lora/lora_baseline_v1/8000"
# "wan_precomp_w1s1_blk20_no_init_bias|pi05_libero_lora_wan_precomp_last_blk|s3://behavior-challenge/openpi_checkpoints/pi05_libero_lora_wan_precomp/wan_precomp_v1_w1s1_blk20_NO_INIT_BIAS/29999"
# "semonly|pi05_libero_lora_wan_precomp_semonly|s3://behavior-challenge/openpi_checkpoints/pi05_libero_lora_wan_precomp_semonly/wan_precomp_semonly_v1/8000"
# "baseline_full_finetune_bs_128|pi05_libero_wan_precomp|s3://behavior-challenge/openpi_checkpoints/pi05_libero_wan_precomp/libero_wan_full_finetune/6000"
# "golden_boy_baseline_bs32_deeplora_ki_ar|pi05_libero_deeplora_ki_ar|s3://behavior-challenge/openpi_checkpoints/pi05_libero_deeplora_ki_ar/libero_deeplora_ki_ar_2nd/17999"
# "baseline_ki_ar_bs_256_NO_WAN_DESPITE_CONFIG_NAME|pi05_libero_deeplora_ki_ar|s3://behavior-challenge/openpi_checkpoints/pi05_libero_deeplora_ki_ar_wan_precomp/libero_deeplora_ki_ar_wan_precomp/3000"
# "golden_boy_bs32_deeplora_ki_ar_WITH_WAN|pi05_libero_deeplora_ki_ar_wan_precomp|s3://behavior-challenge/openpi_checkpoints/pi05_libero_deeplora_ki_ar_wan_precomp/libero_deeplora_ki_ar_wan_precomp_fr/17999"
# "gatewarmup_step9000|pi05_libero_deeplora_wan_precomp_gatewarmup|s3://behavior-challenge/openpi_checkpoints/pi05_libero_deeplora_wan_precomp_gatewarmup/libero_deeplora_wan_precomp_gatewarmup/9000"
# "fft_baseline_step8000|pi05_libero_fft|s3://behavior-challenge/openpi_checkpoints/pi05_libero_fft/libero_fft/8000"
# "fft_baseline_step4000|pi05_libero_fft|s3://behavior-challenge/openpi_checkpoints/pi05_libero_fft/libero_fft/4000"
# "fft_baseline_step15000|pi05_libero_fft|s3://behavior-challenge/openpi_checkpoints/pi05_libero_fft/libero_fft/15000"

# "libero_fft_smallbatch_step29999|pi05_libero_fft_smallbatch|s3://behavior-challenge/openpi_checkpoints/pi05_libero_fft_smallbatch/libero_fft_smallbatch/29999"
# "fft_wan_gatewarmup_step15000|pi05_libero_fft_wan_precomp_gatewarmup|s3://behavior-challenge/openpi_checkpoints/pi05_libero_fft_wan_precomp_gatewarmup/fft_wan_precomp_gatewarmup/15000"

RUNS=(
    "wan_fft_warmup_smallbatch_step20000|pi05_libero_fft_wan_precomp_gatewarmup_smallbatch|s3://behavior-challenge/openpi_checkpoints/pi05_libero_fft_wan_precomp_gatewarmup_smallbatch/fft_wan_precomp_gatewarmup_smallbatch/20000"
)

SUITES=(
    "libero_spatial_swap"
    "libero_object_swap"
    "libero_goal_swap"
    "libero_10_swap"
    "libero_spatial_lan"
    "libero_object_lan"
    "libero_goal_lan"
    "libero_10_lan"
    "libero_spatial_object"
    "libero_object_object"
    "libero_goal_object"
    "libero_10_object"
    "libero_spatial"
    "libero_object"
    "libero_goal"
    "libero_10"
    "libero_spatial_task"
    "libero_object_task"
    "libero_goal_task"
    "libero_10_task"
)

EXTRA_EVAL_ARGS=("$@")

OUT_ROOT="${LIBERO_SWEEP_OUT_ROOT:-$SCRIPT_DIR/../data/libero_sweep}"
mkdir -p "$OUT_ROOT"
SUMMARY_LOG="$OUT_ROOT/sequence_$(date +%Y%m%d_%H%M%S).log"
echo "[sequence_libero_evals] writing summary to $SUMMARY_LOG"

overall_rc=0
for entry in "${RUNS[@]}"; do
    IFS='|' read -r LABEL CONFIG CKPT <<<"$entry"
    for SUITE in "${SUITES[@]}"; do
        echo "============================================================"
        echo "[sequence_libero_evals] $(date) label=$LABEL suite=$SUITE"
        echo "  config=$CONFIG"
        echo "  ckpt=$CKPT"
        echo "============================================================"

        VIDEO_DIR="$OUT_ROOT/$LABEL/videos"
        mkdir -p "$VIDEO_DIR"

        bash "$RUN_SCRIPT" \
            --config "$CONFIG" \
            --checkpoint "$CKPT" \
            -- \
            --args.task-suite-name "$SUITE" \
            --args.video-out-path "$VIDEO_DIR" \
            "${EXTRA_EVAL_ARGS[@]}"
        rc=$?

        echo "[sequence_libero_evals] $(date) label=$LABEL suite=$SUITE rc=$rc" | tee -a "$SUMMARY_LOG"
        if [[ $rc -ne 0 ]]; then
            overall_rc=$rc
        fi
    done
done

echo "[sequence_libero_evals] done. overall_rc=$overall_rc (see $SUMMARY_LOG)"
exit $overall_rc
