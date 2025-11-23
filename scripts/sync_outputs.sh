#!/bin/bash
# Sync outputs from Vast.ai instance to local machine or S3
# Usage: ./sync_outputs.sh [instance_host] [local_dir] [remote_dir]

set -e

INSTANCE_HOST="${1:-vastai-instance-1}"
LOCAL_DIR="${2:-./outputs}"
REMOTE_DIR="${3:-/workspace/persistent}"

echo "=========================================="
echo "Syncing outputs from Vast.ai instance"
echo "=========================================="
echo "Instance: $INSTANCE_HOST"
echo "Remote: $REMOTE_DIR"
echo "Local: $LOCAL_DIR"
echo ""

# Create local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Sync checkpoints
echo "Syncing checkpoints..."
rsync -avz --progress \
    "${INSTANCE_HOST}:${REMOTE_DIR}/checkpoints/" \
    "${LOCAL_DIR}/checkpoints/"

# Sync logs
echo "Syncing logs..."
rsync -avz --progress \
    "${INSTANCE_HOST}:${REMOTE_DIR}/../logs/" \
    "${LOCAL_DIR}/logs/"

# Sync experiment outputs
if ssh "$INSTANCE_HOST" "[ -d ${REMOTE_DIR}/experiments ]"; then
    echo "Syncing experiment outputs..."
    rsync -avz --progress \
        "${INSTANCE_HOST}:${REMOTE_DIR}/experiments/" \
        "${LOCAL_DIR}/experiments/"
fi

echo ""
echo "=========================================="
echo "Sync complete!"
echo "=========================================="

# Optional: Upload to S3 if configured
if [ -n "$S3_BUCKET" ]; then
    echo ""
    echo "Uploading to S3..."
    aws s3 sync "${LOCAL_DIR}/checkpoints/" \
        "s3://${S3_BUCKET}/checkpoints/" \
        --exclude "*.pt" --include "checkpoint_*.pt"
    echo "S3 upload complete!"
fi

