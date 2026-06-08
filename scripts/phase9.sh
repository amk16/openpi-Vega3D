#!/usr/bin/env bash
# =============================================================================
# Phase 9 runbook — fidelity-fixed FFT+WAN: the discriminating experiment.
# =============================================================================
# Runs the Phase-8 fidelity fixes (branch `fidelity-fixes`) end-to-end on a
# rented GPU box, STAGE BY STAGE with fail-fast gates so a wiring bug or a
# missing credential is caught in minutes, not after a 15-hour job.
#
# Usage (run each stage deliberately; inspect output before the next):
#   bash scripts/phase9.sh setup     # 0. env + creds preflight
#   bash scripts/phase9.sh verify     # 1. GATE: pytest suite (minutes)
#   bash scripts/phase9.sh before     # 2. GATE: before-evidence diagnostics
#   bash scripts/phase9.sh dryrun     # 3. GATE: 5-episode pipeline smoke
#   bash scripts/phase9.sh regen      # 4. full _cpool cache regen (long, GPU)
#   bash scripts/phase9.sh train      # 5. fidelityfix training run (longest)
#   bash scripts/phase9.sh eval       # 6. 4 swap suites (separate LIBERO env)
#   bash scripts/phase9.sh status     # show where things are
#
# ALWAYS run the long stages (regen/train) inside tmux:  tmux new -s p9
#
# >>> FILL THESE IN before running (and `export AWS_*`, `HF_TOKEN` in your shell):
EXP_NAME="${EXP_NAME:-fidelityfix_v1}"
FF_CONFIG="pi05_libero_fft_wan_precomp_gatewarmup_fidelityfix"
HEADLINE_CONFIG="pi05_libero_fft_wan_precomp_gatewarmup"
S3_BUCKET="behavior-challenge"
# Legacy (pre-fix) cache S3 prefix — the before-evidence source. Derived from
# the headline config's variant_tag; override if your bucket layout differs:
LEGACY_CACHE_S3="s3://${S3_BUCKET}/tower_features/physical-intelligence_libero/wan_t2v_16x1536_w1s1_blk20"
LEGACY_CACHE_LOCAL="tower_features/physical-intelligence_libero/wan_t2v_16x1536_w1s1_blk20"
# A PUBLISHED 35.3% checkpoint dir (s3://...) for the norm-ratio diagnostic.
# <<< REQUIRED for `before`: paste the FFT+WAN gatewarmup checkpoint path.
PUBLISHED_CKPT_S3="${PUBLISHED_CKPT_S3:-}"
# WAN tower checkpoint dir (auto-downloaded by precompute if missing):
WAN_CKPT_DIR="${WAN_T2V_CKPT_DIR:-/workspace/openpi-Vega3D/ckpts/Wan2.1-T2V-1.3B}"
REGEN_BATCH="${REGEN_BATCH:-16}"   # precompute default is 4; bump to cut wall-time
# =============================================================================

set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root
STAGE="${1:-}"

say()  { printf '\n\033[1;36m[phase9:%s]\033[0m %s\n' "$STAGE" "$*"; }
die()  { printf '\n\033[1;31m[phase9:%s] FAIL:\033[0m %s\n' "$STAGE" "$*" >&2; exit 1; }
gate() { printf '\n\033[1;33m================ GATE: %s ================\033[0m\n' "$*"; }

require_cmd() { command -v "$1" >/dev/null 2>&1 || die "missing command: $1"; }
require_env() { [[ -n "${!1:-}" ]] || die "env var $1 is not set"; }

case "$STAGE" in

setup)
    say "Preflight: tools, GPU, credentials."
    require_cmd uv; require_cmd aws; require_cmd git
    git rev-parse --abbrev-ref HEAD | grep -qx fidelity-fixes \
        || die "not on branch fidelity-fixes (got: $(git rev-parse --abbrev-ref HEAD))"
    require_env AWS_ACCESS_KEY_ID; require_env AWS_SECRET_ACCESS_KEY
    [[ -n "${HF_TOKEN:-}" ]] || say "WARN: HF_TOKEN unset — WAN/dataset download may fail if gated."
    say "Installing deps (GIT_LFS_SKIP_SMUDGE=1 uv sync)…"
    GIT_LFS_SKIP_SMUDGE=1 uv sync || die "uv sync failed"
    GIT_LFS_SKIP_SMUDGE=1 uv pip install -e . || die "editable install failed"
    if command -v nvidia-smi >/dev/null 2>&1; then nvidia-smi || true; else say "WARN: no nvidia-smi — GPU not visible?"; fi
    df -h . | tail -1
    say "Setup OK. Next: bash scripts/phase9.sh verify"
    ;;

verify)
    gate "1 — code verification (must pass before any GPU-hours)"
    say "Running the CI-safe pytest suite…"
    uv run pytest --strict-markers -m "not manual" || die "pytest failed — FIX before proceeding."
    say "probe_wan contract check (expect (1, 256, 1536))…"
    uv run python scripts/probe_wan.py || say "WARN: probe_wan failed (needs WAN ckpt) — re-run after 'before'."
    if [[ -d "$WAN_CKPT_DIR" ]]; then
        say "Running the manual full-encoder geometry test…"
        WAN_T2V_CKPT_DIR="$WAN_CKPT_DIR" uv run pytest scripts/test_tower.py -m manual || die "manual encoder test failed."
    fi
    say "GATE 1 PASSED. Next: bash scripts/phase9.sh before"
    ;;

before)
    gate "2 — before-evidence diagnostics (on the EXISTING cache, BEFORE regen)"
    say "Syncing a slice of the legacy cache from $LEGACY_CACHE_S3 …"
    mkdir -p "$LEGACY_CACHE_LOCAL"
    aws s3 sync "$LEGACY_CACHE_S3" "$LEGACY_CACHE_LOCAL" \
        --exclude "*" --include "meta.json" \
        --include "*/ep_000000.safetensors" --include "*/ep_000001.safetensors" \
        --include "*/ep_000002.safetensors" --include "*/ep_000003.safetensors" \
        --include "*/ep_000004.safetensors" || die "legacy cache sync failed (check LEGACY_CACHE_S3)."
    say "Break-1 signature (expect weak energy in the ~3 outermost columns each side):"
    uv run python scripts/diagnose_wan_fidelity.py column-energy \
        --cache_dir "$LEGACY_CACHE_LOCAL" --episodes 5 || die "column-energy failed."
    if [[ -n "$PUBLISHED_CKPT_S3" ]]; then
        say "Break-2 signature (expect ||f_gen||/||f_sem|| far from 1):"
        local_ckpt="checkpoints/published_norm_ratio"
        mkdir -p "$local_ckpt"
        aws s3 sync "$PUBLISHED_CKPT_S3" "$local_ckpt" || die "checkpoint sync failed."
        uv run python scripts/diagnose_wan_fidelity.py norm-ratio \
            --config "$HEADLINE_CONFIG" --checkpoint "$local_ckpt" || die "norm-ratio failed."
    else
        say "WARN: PUBLISHED_CKPT_S3 unset — skipping Break-2 norm-ratio. Set it to run the full before-evidence."
    fi
    gate "STOP & READ: if the Break-1 columns are NOT visibly weak at the edges"
    say "(or the Break-2 ratio is ~1), PAUSE — the breaks may not be biting as the audit predicted."
    say "If the signatures are present: Next: bash scripts/phase9.sh dryrun"
    ;;

dryrun)
    gate "3 — pipeline smoke (5 episodes, local-only, ~20 min)"
    say "Regen 5 episodes into the _cpool cache (no S3)…"
    uv run python scripts/precompute_tower_features.py "$FF_CONFIG" \
        --window 1 --limit_episodes 5 --s3_bucket "" || die "precompute smoke failed."
    cpool_dir="tower_features/physical-intelligence_libero/wan_t2v_16x1536_w1s1_blk20_cpool"
    [[ -f "$cpool_dir/meta.json" ]] || die "no meta.json in $cpool_dir — cache namespace wrong?"
    grep -q '"content_region_pool": true' "$cpool_dir/meta.json" \
        || die "meta.json missing content_region_pool:true — flag not wired into the cache."
    say "Cache namespace + flag provenance OK."
    say "GATE 3 PASSED. Next (commit GPU-hours): bash scripts/phase9.sh regen"
    ;;

regen)
    say "FULL cache regen → _cpool (long; run inside tmux). batch=$REGEN_BATCH"
    require_env AWS_ACCESS_KEY_ID
    uv run python scripts/precompute_tower_features.py "$FF_CONFIG" \
        --window 1 --batch_size "$REGEN_BATCH" --s3_bucket "$S3_BUCKET" || die "regen failed."
    say "Regen done. Legacy cache/prefix untouched. Next: bash scripts/phase9.sh train"
    ;;

train)
    say "Fidelity-fixed training run (longest; tmux). config=$FF_CONFIG exp=$EXP_NAME"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py \
        "$FF_CONFIG" --exp-name="$EXP_NAME" --overwrite || die "training failed."
    say "Training done. Checkpoints under s3://$S3_BUCKET/.../$FF_CONFIG/$EXP_NAME/"
    say "Next (separate LIBERO env): bash scripts/phase9.sh eval"
    ;;

eval)
    gate "6 — eval the 4 swap suites @ 50 ep/task"
    say "Edit scripts/sequence_libero_evals.sh RUNS to point at the fidelityfix checkpoint, then:"
    cat <<'EOF'
  # In sequence_libero_evals.sh, set RUNS to a single entry, e.g.:
  #   RUNS=( "fidelityfix|pi05_libero_fft_wan_precomp_gatewarmup_fidelityfix|s3://behavior-challenge/openpi_checkpoints/pi05_libero_fft_wan_precomp_gatewarmup_fidelityfix/fidelityfix_v1/<step>" )
  # and SUITES to the four LIBERO-Pro swap suites:
  #   SUITES=( libero_spatial_swap libero_object_swap libero_goal_swap libero_10_swap )
  # then:
  bash scripts/sequence_libero_evals.sh --args.num-trials-per-task 50
EOF
    say "Decision rule: <=42% -> 'genuinely not useful' earned; meaningfully >35.3% -> fidelity story was real."
    ;;

status)
    say "Branch: $(git rev-parse --abbrev-ref HEAD)"
    say "Configs present:"; uv run python -c "import openpi.training.config as c; print('$FF_CONFIG' in c._CONFIGS_DICT and 'fidelityfix config OK' or 'MISSING')" 2>/dev/null || true
    ls -d tower_features/*/*cpool 2>/dev/null && say "regen cache present" || say "no _cpool cache yet"
    ;;

*)
    grep '^#' "$0" | sed 's/^# \{0,1\}//' | head -30
    die "unknown or missing stage. Pick one: setup|verify|before|dryrun|regen|train|eval|status"
    ;;
esac
