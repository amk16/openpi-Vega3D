#!/usr/bin/env bash
#
# Run a LIBERO (or LIBERO-Pro) eval end-to-end:
#   1. start the pi05 inference websocket server in the background (main venv)
#   2. wait for it to bind the port
#   3. run examples/libero/main.py against it (libero venv)
#   4. always stop the server on exit / Ctrl-C / failure
#
# Designed to run inside the Vast.ai / Runpod Docker image — no docker-in-docker,
# no docker compose; both processes are plain python procs in the same container.
#
# Usage:
#   bash scripts/run_libero_eval.sh \
#       --config pi05_libero_lora_wan_precomp \
#       --checkpoint s3://behavior-challenge/checkpoints/pi05_libero_lora_wan_precomp/exp/30000 \
#       [--port 8000] \
#       [--server-timeout 600] \
#       [--ckpt-cache-dir checkpoints] \
#       -- \
#       --args.task-suite-name libero_spatial \
#       --args.num-trials-per-task 10 \
#       --args.video-out-path data/libero/videos
#
# Anything after `--` is forwarded verbatim to examples/libero/main.py.
#
# Checkpoint resolution:
#   - s3://...  → `aws s3 sync` into $CKPT_CACHE_DIR/<bucket>/<key>, then use that
#                 local path. `aws s3 sync` is incremental, so re-running with the
#                 same checkpoint is cheap.
#   - gs://...  → passed through unchanged (serve_policy.py loads gs:// natively).
#   - anything else → treated as a local path.

set -eo pipefail

usage() {
    sed -n '3,28p' "$0"
}

CONFIG=""
CHECKPOINT=""
PORT=8000
SERVER_TIMEOUT=600
CKPT_CACHE_DIR="${LIBERO_CKPT_CACHE_DIR:-checkpoints}"
EVAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)          CONFIG="$2"; shift 2 ;;
        --checkpoint)      CHECKPOINT="$2"; shift 2 ;;
        --port)            PORT="$2"; shift 2 ;;
        --server-timeout)  SERVER_TIMEOUT="$2"; shift 2 ;;
        --ckpt-cache-dir)  CKPT_CACHE_DIR="$2"; shift 2 ;;
        -h|--help)         usage; exit 0 ;;
        --)                shift; EVAL_ARGS+=("$@"); break ;;
        *)                 EVAL_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "$CONFIG" || -z "$CHECKPOINT" ]]; then
    echo "Error: --config and --checkpoint are required." >&2
    usage >&2
    exit 1
fi

MAIN_VENV="/venv/main"
LIBERO_VENV="/venv/libero"
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

for v in "$MAIN_VENV" "$LIBERO_VENV"; do
    if [[ ! -d "$v" ]]; then
        echo "Error: venv not found at $v" >&2
        echo "Are you running inside the Vast.ai / Runpod Docker image?" >&2
        exit 1
    fi
done

# --------- Resolve checkpoint (sync from S3 if needed) ----------
if [[ "$CHECKPOINT" == s3://* ]]; then
    if ! command -v aws >/dev/null 2>&1; then
        echo "Error: --checkpoint is an s3:// URI but the 'aws' CLI is not on PATH." >&2
        exit 1
    fi
    # Mirror the s3 path under the cache dir so different checkpoints don't collide.
    REMOTE_PATH="${CHECKPOINT#s3://}"
    LOCAL_CKPT="$CKPT_CACHE_DIR/$REMOTE_PATH"
    mkdir -p "$LOCAL_CKPT"
    echo "[run_libero_eval] Syncing checkpoint $CHECKPOINT -> $LOCAL_CKPT"
    aws s3 sync "$CHECKPOINT" "$LOCAL_CKPT"
    CHECKPOINT="$LOCAL_CKPT"
elif [[ "$CHECKPOINT" == gs://* ]]; then
    echo "[run_libero_eval] Using gs:// checkpoint directly (loaded by serve_policy.py)"
else
    if [[ ! -d "$CHECKPOINT" ]]; then
        echo "Error: local checkpoint dir does not exist: $CHECKPOINT" >&2
        exit 1
    fi
fi

LOG_DIR="${LIBERO_EVAL_LOG_DIR:-$SCRIPT_DIR/data/libero/server_logs}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SAFE_CONFIG="${CONFIG//\//_}"
SERVER_LOG="$LOG_DIR/serve_policy_${SAFE_CONFIG}_${TS}.log"

SERVER_PID=""
cleanup() {
    local rc=$?
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run_libero_eval] Stopping inference server (pid=$SERVER_PID)"
        kill -TERM "$SERVER_PID" 2>/dev/null || true
        # Give it ~10s to exit gracefully, then SIGKILL.
        for _ in $(seq 1 20); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 0.5
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "[run_libero_eval] Server did not exit; sending SIGKILL"
            kill -KILL "$SERVER_PID" 2>/dev/null || true
        fi
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    exit $rc
}
trap cleanup EXIT INT TERM

echo "[run_libero_eval] config         = $CONFIG"
echo "[run_libero_eval] checkpoint     = $CHECKPOINT"
echo "[run_libero_eval] port           = $PORT"
echo "[run_libero_eval] ckpt cache dir = $CKPT_CACHE_DIR"
echo "[run_libero_eval] server log     = $SERVER_LOG"

# --------- 1. Launch inference server in the background ----------
(
    # shellcheck disable=SC1091
    source "$MAIN_VENV/bin/activate"
    export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/packages/openpi-client/src:${PYTHONPATH:-}"
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 exec python "$SCRIPT_DIR/scripts/serve_policy.py" \
        --port "$PORT" \
        --load-live-tower \
        policy:checkpoint \
        --policy.config "$CONFIG" \
        --policy.dir "$CHECKPOINT"
) >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[run_libero_eval] Inference server pid=$SERVER_PID"

# --------- 2. Wait for the server to accept connections ----------
echo "[run_libero_eval] Waiting up to ${SERVER_TIMEOUT}s for server on 127.0.0.1:${PORT}..."
ready=0
for i in $(seq 1 "$SERVER_TIMEOUT"); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[run_libero_eval] Server process exited before opening port. Tail of log:" >&2
        tail -n 50 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
        exec 3<&- 3>&-
        ready=1
        echo "[run_libero_eval] Server is up after ~${i}s"
        break
    fi
    sleep 1
done
if [[ $ready -ne 1 ]]; then
    echo "[run_libero_eval] Server did not become ready within ${SERVER_TIMEOUT}s. Tail of log:" >&2
    tail -n 50 "$SERVER_LOG" >&2 || true
    exit 1
fi

# --------- 3. Run the LIBERO eval ----------
echo "[run_libero_eval] Starting LIBERO eval"
set +e
(
    # shellcheck disable=SC1091
    source "$LIBERO_VENV/bin/activate"
    export PYTHONPATH="$SCRIPT_DIR:$SCRIPT_DIR/packages/openpi-client/src:${PYTHONPATH:-}"
    export LIBERO_CONFIG_PATH=/opt/libero-config
    export MUJOCO_GL=egl
    export PYOPENGL_PLATFORM=egl
    exec python "$SCRIPT_DIR/examples/libero/main.py" \
        --args.host 127.0.0.1 \
        --args.port "$PORT" \
        "${EVAL_ARGS[@]}"
)
EVAL_RC=$?
set -e

echo "[run_libero_eval] Eval exited with code $EVAL_RC"
# trap cleanup will stop the server and propagate EVAL_RC via exit
exit $EVAL_RC
