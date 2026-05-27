#!/usr/bin/env bash

set -euo pipefail

# Array of local directories to sync
LOCAL_DIRS=(
  "tower_features/physical-intelligence_libero/cosmos_policy_libero_16x2048_w1s1_blk20_t5cond_gpu0/"
  "tower_features/physical-intelligence_libero/cosmos_policy_libero_16x2048_w1s1_blk20_t5cond_gpu1/"
)

# S3 bucket base (edit here if bucket or main prefix changes)
S3_BASE="s3://behavior-challenge"

# Array of S3 destinations corresponding to the local directories
S3_URIS=(
  "${S3_BASE}/tower_features/physical-intelligence_libero/cosmos_policy_libero_16x2048_w1s1_blk20_t5cond/"
  "${S3_BASE}/tower_features/physical-intelligence_libero/cosmos_policy_libero_16x2048_w1s1_blk20_t5cond/"
)

# How often to sync (in seconds)
INTERVAL_SECONDS="${INTERVAL_SECONDS:-5}" # 70 seconds

if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI not found in PATH." >&2
  exit 1
fi

for LOCAL_DIR in "${LOCAL_DIRS[@]}"; do
  if [[ ! -d "${LOCAL_DIR}" ]]; then
    echo "Error: local directory '${LOCAL_DIR}' does not exist." >&2
    exit 1
  fi
done

echo "Starting periodic sync of directories:"
for idx in "${!LOCAL_DIRS[@]}"; do
  echo "  local: ${LOCAL_DIRS[$idx]}"
  echo "  s3:    ${S3_URIS[$idx]}"
done
echo "  every: ${INTERVAL_SECONDS}s"
echo

while true; do
  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Sync started"

  all_success=true
  for idx in "${!LOCAL_DIRS[@]}"; do
    local_dir="${LOCAL_DIRS[$idx]}"
    s3_uri="${S3_URIS[$idx]}"
    echo "  Syncing: ${local_dir} -> ${s3_uri}"
    if aws s3 sync "${local_dir}" "${s3_uri}"; then
      echo "    [$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Directory sync finished successfully"
    else
      exit_code=$?
      echo "    [$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Directory sync failed (exit ${exit_code}); will retry after sleep" >&2
      all_success=false
    fi
  done

  if [ "${all_success}" = true ]; then
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] All syncs finished successfully"
  else
    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Some syncs failed; see above. Will retry."
  fi

  echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Sleeping ${INTERVAL_SECONDS}s"
  sleep "${INTERVAL_SECONDS}"
done
