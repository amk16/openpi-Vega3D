#!/usr/bin/env bash

set -euo pipefail

# Set the local directory to sync (edit here if needed)
LOCAL_DIR="tower_features/physical-intelligence_libero/wan_t2v_16x1536_w1s1_blk20"

# S3 bucket base (edit here if bucket or main prefix changes)
S3_BASE="s3://behavior-challenge"

# Derive S3_URI from LOCAL_DIR (sync path preserved after bucket base)
S3_URI="${S3_BASE}/${LOCAL_DIR}"

# How often to sync (in seconds)
INTERVAL_SECONDS="${INTERVAL_SECONDS:-70}" # 70 seconds

if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI not found in PATH." >&2
  exit 1
fi

if [[ ! -d "${LOCAL_DIR}" ]]; then
  echo "Error: local directory '${LOCAL_DIR}' does not exist." >&2
  exit 1
fi

echo "Starting periodic sync:"
echo "  local: ${LOCAL_DIR}"
echo "  s3:    ${S3_URI}"
echo "  every: ${INTERVAL_SECONDS}s"
echo

while true; do
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Sync started"

  # This is the command that emits logs like:
  # upload: <local_file> to s3://<bucket>/<key>
  if aws s3 sync "${LOCAL_DIR}" "${S3_URI}"; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Sync finished successfully"
  else
    exit_code=$?
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Sync failed (exit ${exit_code}); will retry after sleep" >&2
  fi

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Sleeping ${INTERVAL_SECONDS}s"
  sleep "${INTERVAL_SECONDS}"
done
