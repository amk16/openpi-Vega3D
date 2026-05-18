#!/usr/bin/env bash

set -euo pipefail

# Defaults can be overridden with args:
#   ./scripts/sync_tower_features_to_s3.sh [LOCAL_DIR] [S3_URI]
LOCAL_DIR="${1:-tower_features}"
S3_URI="${2:-s3://behavior-challenge/tower_features}"
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
