#!/usr/bin/env bash
set -euo pipefail

# Move S3 objects one-by-one from source prefix to destination prefix.
# By default, this moves:
#   s3://behavior-challenge/checkpoints/openpi_checkpoints/
# -> s3://behavior-challenge/openpi_checkpoints/
#
# Usage:
#   scripts/move_s3_objects_individually.sh [--dry-run]
#
# Notes:
# - This script does NOT use `aws s3 sync`.
# - It lists all object keys under the source prefix first, then moves each key
#   individually with `aws s3 mv`.

BUCKET="behavior-challenge"
SRC_PREFIX="checkpoints/openpi_checkpoints/"
DST_PREFIX="openpi_checkpoints/"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "Error: aws CLI not found in PATH."
  exit 1
fi

if [[ "$SRC_PREFIX" == "$DST_PREFIX" ]]; then
  echo "Error: source and destination prefixes are identical."
  exit 1
fi

keys_file="$(mktemp)"
trap 'rm -f "$keys_file"' EXIT

echo "Listing all objects under s3://${BUCKET}/${SRC_PREFIX} ..."
aws s3api list-objects-v2 \
  --bucket "$BUCKET" \
  --prefix "$SRC_PREFIX" \
  --query 'Contents[].Key' \
  --output text | tr '\t' '\n' | sed '/^None$/d' > "$keys_file"

if [[ ! -s "$keys_file" ]]; then
  echo "No objects found under s3://${BUCKET}/${SRC_PREFIX}"
  exit 0
fi

total="$(wc -l < "$keys_file" | tr -d ' ')"
echo "Found ${total} objects."
echo

count=0
while IFS= read -r src_key; do
  [[ -z "$src_key" ]] && continue
  count=$((count + 1))

  if [[ "$src_key" != "$SRC_PREFIX"* ]]; then
    echo "Skipping unexpected key: $src_key"
    continue
  fi

  dst_key="${src_key/#$SRC_PREFIX/$DST_PREFIX}"
  src_uri="s3://${BUCKET}/${src_key}"
  dst_uri="s3://${BUCKET}/${dst_key}"

  echo "[${count}/${total}] ${src_uri} -> ${dst_uri}"
  if [[ "$DRY_RUN" == "true" ]]; then
    continue
  fi

  aws s3 mv "$src_uri" "$dst_uri"
done < "$keys_file"

if [[ "$DRY_RUN" == "true" ]]; then
  echo
  echo "Dry run complete. No files were moved."
else
  echo
  echo "Move complete."
fi
