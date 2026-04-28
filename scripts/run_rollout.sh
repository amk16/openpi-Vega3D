#!/bin/bash
# =============================================================================
# openpi-Vega3D BEHAVIOR Rollout Launcher
# =============================================================================
# Sets up all environment variables required by Isaac Sim / OmniGibson /
# BEHAVIOR and then runs the rollout script.
#
# Uses RLinf's .venv-openpi as the Python environment. This is the same venv
# that RLinf's minimal_rollout.sh uses -- it has torch, omnigibson, isaacsim,
# and the Kit runtime (omni) all installed coherently in one environment.
# openpi-Vega3D source is added via PYTHONPATH (same pattern as openpi/src
# in minimal_rollout.sh).
#
# WHY .venv-openpi instead of openpi-Vega3D/.venv:
#   The openpi-Vega3D/.venv lacked isaacsim and omni. Previous approach grafted
#   ALL of RLinf's .venv site-packages onto PYTHONPATH, causing hundreds of
#   duplicate packages and ABI conflicts that segfaulted when PyTorch and Kit
#   initialized CUDA in the same process. Using .venv-openpi avoids this
#   entirely -- one coherent package set, no PYTHONPATH pollution.
#
# Usage:
#   bash scripts/run_rollout.sh --task_name turning_on_radio --ckpt_dir /path/to/ckpt
#   bash scripts/run_rollout.sh --help   # See all available flags
#
# Override paths via environment variables before calling:
#   VENV_OPENPI_OVERRIDE  -- custom .venv-openpi location
#   ISAAC_PATH_OVERRIDE   -- custom Isaac Sim location
#   BEHAVIOR_DATA_OVERRIDE -- custom BEHAVIOR-1K datasets path
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Activate .venv-openpi ────────────────────────────────────────────────────

VENV_OPENPI="${VENV_OPENPI_OVERRIDE:-/workspace/RLinf/.venv-openpi}"

if [ -z "${VIRTUAL_ENV:-}" ]; then
    if [ -f "$VENV_OPENPI/bin/activate" ]; then
        source "$VENV_OPENPI/bin/activate"
    else
        echo "ERROR: .venv-openpi not found at $VENV_OPENPI"
        echo "       Set VENV_OPENPI_OVERRIDE or check RLinf setup."
        exit 1
    fi
fi

# ── EULA and headless flags ──────────────────────────────────────────────────

export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
export OMNIGIBSON_NO_SIGNALS=1
export OMNIGIBSON_NO_OMNI_LOGS=1

# ── Isaac Sim paths ──────────────────────────────────────────────────────────
# isaacsim and omni are installed directly in .venv-openpi's site-packages.

SITE_PACKAGES="$VENV_OPENPI/lib/python3.10/site-packages"
ISAAC_PATH="${ISAAC_PATH_OVERRIDE:-$SITE_PACKAGES/isaacsim}"
export ISAAC_PATH
export EXP_PATH="$ISAAC_PATH/apps"

CARB_APP_PATH="${SITE_PACKAGES}/omni"
export CARB_APP_PATH

if [ ! -d "$ISAAC_PATH" ]; then
    echo "WARNING: ISAAC_PATH does not exist: $ISAAC_PATH"
fi

# ── BEHAVIOR assets ──────────────────────────────────────────────────────────

export OMNIGIBSON_DATA_PATH="${BEHAVIOR_DATA_OVERRIDE:-/workspace/BEHAVIOR-1K/datasets}"

if [ ! -d "$OMNIGIBSON_DATA_PATH" ]; then
    echo "WARNING: OMNIGIBSON_DATA_PATH does not exist: $OMNIGIBSON_DATA_PATH"
fi

# ── Rendering (headless GPU) ─────────────────────────────────────────────────

export VK_DRIVER_FILES="${VK_DRIVER_FILES:-/etc/vulkan/icd.d/nvidia_icd.json}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# Disable torch.compile / Inductor during rollout -- the JIT compilation overhead
# is wasteful for inference and can conflict with Kit's CUDA context.
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

# ── Python path ──────────────────────────────────────────────────────────────
# Only source directories -- no site-packages grafting.
# .venv-openpi already has all packages installed; PYTHONPATH adds source trees
# for RLinf (rlinf module) and openpi-Vega3D (openpi_vega3d + openpi modules).

export PYTHONPATH="$REPO_ROOT/src:/workspace/RLinf:${PYTHONPATH:-}"

# ── Print config ─────────────────────────────────────────────────────────────

echo "============================================"
echo "  openpi-Vega3D BEHAVIOR Rollout"
echo "============================================"
echo "  Venv:                $VENV_OPENPI"
echo "  Python:              $(which python)"
echo "  Python version:      $(python --version 2>&1)"
echo "  ISAAC_PATH:          $ISAAC_PATH"
echo "  CARB_APP_PATH:       $CARB_APP_PATH"
echo "  EXP_PATH:            $EXP_PATH"
echo "  OMNIGIBSON_DATA_PATH: $OMNIGIBSON_DATA_PATH"
echo "  PYTHONPATH:          $PYTHONPATH"
echo "  REPO_ROOT:           $REPO_ROOT"
echo "============================================"

ROLLOUT_LOG_DIR="${ROLLOUT_LOG_DIR:-$REPO_ROOT/outputs/rollouts}"
export ROLLOUT_LOG_DIR
mkdir -p "$ROLLOUT_LOG_DIR"
echo "  ROLLOUT_LOG_DIR:     $ROLLOUT_LOG_DIR"
echo "============================================"
echo ""

# ── Run ──────────────────────────────────────────────────────────────────────

exec python "$REPO_ROOT/scripts/run_rollout.py" "$@"
