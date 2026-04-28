#!/bin/bash
# =============================================================================
# openpi-Vega3D Environment Setup
# =============================================================================
# Creates a Python 3.10 venv and installs all dependencies needed for
# BEHAVIOR rollouts with the Pi05 policy + VEGA-3D generative towers.
#
# Prerequisites (already on this machine from RLinf setup):
#   - /workspace/RLinf/.venv       -- Isaac Sim venv (referenced via ISAAC_PATH)
#   - /workspace/BEHAVIOR-1K/      -- OmniGibson, BDDL, datasets
#   - ~/.cache/openpi/big_vision/  -- PaliGemma tokenizer
#
# Usage:
#   bash scripts/setup_env.sh               # Standard install
#   bash scripts/setup_env.sh --install-joylo  # Also install gello/joylo
#
# This creates .venv/ in the openpi-Vega3D repo root.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BEHAVIOR_1K_DIR="/workspace/BEHAVIOR-1K"
INSTALL_JOYLO=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-joylo)
            INSTALL_JOYLO=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash scripts/setup_env.sh [--install-joylo]"
            exit 1
            ;;
    esac
done

cd "$REPO_ROOT"

# ── 1. Verify prerequisites ──────────────────────────────────────────────────

echo "============================================"
echo "  openpi-Vega3D Environment Setup"
echo "============================================"

if ! command -v uv &> /dev/null; then
    echo "ERROR: uv not found. Install with: pip install uv"
    exit 1
fi

if [ ! -d "$BEHAVIOR_1K_DIR/OmniGibson" ]; then
    echo "ERROR: BEHAVIOR-1K not found at $BEHAVIOR_1K_DIR"
    echo "       Run BEHAVIOR-1K/setup.sh first (see RLinf docs)."
    exit 1
fi

echo "  uv:            $(uv --version)"
echo "  BEHAVIOR-1K:   $BEHAVIOR_1K_DIR"
echo "  Install JoyLo: $INSTALL_JOYLO"
echo "============================================"

# ── 2. Create venv (Python 3.10 required by Isaac Sim / OmniGibson) ──────────

PYTHON_VERSION="3.10"

if [ -d ".venv" ]; then
    EXISTING_PY=$(.venv/bin/python --version 2>/dev/null || echo "unknown")
    echo "Existing .venv found ($EXISTING_PY). Reusing."
else
    echo "Creating .venv with Python $PYTHON_VERSION..."
    uv venv --python="$PYTHON_VERSION"
fi

source .venv/bin/activate
echo "Activated: $(python --version) at $(which python)"

# ── 3. Install core dependencies via uv sync ─────────────────────────────────

echo ""
echo "Installing core dependencies (uv sync)..."
GIT_LFS_SKIP_SMUDGE=1 UV_TORCH_BACKEND=auto uv sync --python "$PYTHON_VERSION"

echo "Installing openpi-Vega3D in editable mode..."
GIT_LFS_SKIP_SMUDGE=1 UV_TORCH_BACKEND=auto uv pip install -e .

# ── 4. Install OmniGibson + BDDL from BEHAVIOR-1K ────────────────────────────

echo ""
echo "Installing OmniGibson (editable from BEHAVIOR-1K)..."
uv pip install --prerelease=allow -e "$BEHAVIOR_1K_DIR/OmniGibson"

echo "Installing BDDL (editable from BEHAVIOR-1K)..."
uv pip install -e "$BEHAVIOR_1K_DIR/bddl3"

# ── 5. Optional: Install JoyLo/Gello ─────────────────────────────────────────

if [ "$INSTALL_JOYLO" = true ]; then
    if [ -d "$BEHAVIOR_1K_DIR/joylo" ]; then
        echo "Installing JoyLo/Gello (editable from BEHAVIOR-1K)..."
        uv pip install -e "$BEHAVIOR_1K_DIR/joylo"
    else
        echo "WARNING: JoyLo directory not found at $BEHAVIOR_1K_DIR/joylo"
    fi
fi

# ── 6. Apply transformers patch ───────────────────────────────────────────────
# openpi ships patched versions of some transformers model files (gemma,
# paligemma, siglip). These must overwrite the installed transformers package.

TRANSFORMERS_REPLACE="$REPO_ROOT/src/openpi/models_pytorch/transformers_replace"
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
TRANSFORMERS_DEST="$SITE_PACKAGES/transformers"

if [ -d "$TRANSFORMERS_REPLACE" ] && [ -d "$TRANSFORMERS_DEST" ]; then
    echo ""
    echo "Applying transformers patch..."
    cp -r "$TRANSFORMERS_REPLACE"/* "$TRANSFORMERS_DEST/"
    echo "  Patched: $TRANSFORMERS_DEST"
else
    echo "WARNING: Could not apply transformers patch."
    echo "  Source: $TRANSFORMERS_REPLACE (exists: $([ -d "$TRANSFORMERS_REPLACE" ] && echo yes || echo no))"
    echo "  Dest:   $TRANSFORMERS_DEST (exists: $([ -d "$TRANSFORMERS_DEST" ] && echo yes || echo no))"
fi

# ── 7. Verify key imports ────────────────────────────────────────────────────

echo ""
echo "Verifying key imports..."
FAILED=0

python -c "import torch; print(f'  torch {torch.__version__} (CUDA: {torch.cuda.is_available()})')" || FAILED=1
python -c "import openpi; print(f'  openpi OK')" || FAILED=1
python -c "import omnigibson; print(f'  omnigibson {omnigibson.__version__}')" 2>/dev/null || {
    echo "  omnigibson: import failed (may need ISAAC_PATH set -- this is expected during setup)"
}
python -c "import bddl; print(f'  bddl OK')" || FAILED=1

if [ "$INSTALL_JOYLO" = true ]; then
    python -c "import gello; print(f'  gello OK')" || echo "  gello: import failed"
fi

python -c "import diffusers; print(f'  diffusers {diffusers.__version__}')" || echo "  diffusers: not installed (needed for VAE tower)"

echo ""
if [ $FAILED -eq 0 ]; then
    echo "============================================"
    echo "  Setup complete!"
    echo "============================================"
    echo ""
    echo "Next steps:"
    echo "  1. Run: bash scripts/run_rollout.sh --help"
    echo "  2. Or verify: python scripts/verify_env.py"
    echo ""
else
    echo "============================================"
    echo "  Setup completed with warnings (see above)"
    echo "============================================"
fi
