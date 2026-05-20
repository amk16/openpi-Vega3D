#!/bin/bash
set -e

LIBERO_VENV="/venv/libero"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$LIBERO_VENV" ]; then
    echo "Error: LIBERO venv not found at $LIBERO_VENV"
    echo "Are you running inside the Vast.ai Docker image?"
    exit 1
fi

source "$LIBERO_VENV/bin/activate"
export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/packages/openpi-client/src:$PYTHONPATH"
export LIBERO_CONFIG_PATH=/opt/libero-config
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python "$SCRIPT_DIR/examples/libero/main.py" "$@"
