#!/bin/bash
set -e

export CONTAINER_ID=${CONTAINER_ID:-$(hostname)}

PERSISTENT_DIRS=(
    "$UV_PYTHON_INSTALL_DIR"
    "$UV_CACHE_DIR"
    "$WARP_CACHE_PATH"
    "$WANDB_DIR"
    "$WANDB_CACHE_DIR"
    "$WANDB_ARTIFACT_DIR"
    "$WANDB_DATA_DIR"
    "/mnt/gdrive"
)

# creating all persistent directories
echo "--- Checking persistent directories ---"
for dir in "${PERSISTENT_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

# sync github repo
BRANCH_TO_USE="dev/fbr"
if [ -n "$GIT_BRANCH" ]; then
    BRANCH_TO_USE=$GIT_BRANCH
fi
echo "Syncing latest code from Git..."
cd /app
git switch $BRANCH_TO_USE
git fetch origin $BRANCH_TO_USE
git reset --hard origin/$BRANCH_TO_USE
# update dependencies
uv sync --locked --no-editable --no-dev

# mount Google Drive via Rclone if config is provided
# not used for now
if [ -n "$RCLONE_GDRIVE_CONF" ]; then
    echo "Configuring Rclone..."
    mkdir -p ~/.config/rclone/
    echo "[gdrive]
    type = drive
    token = $RCLONE_GDRIVE_CONF" > ~/.config/rclone/rclone.conf
    mkdir -p /mnt/gdrive
    rclone mount gdrive: /mnt/gdrive --vfs-cache-mode full --daemon

    sleep 2
    if mountpoint -q /mnt/gdrive; then
        echo "GDrive successfully mounted at /mnt/gdrive"
    else
        echo "Warning: GDrive mount failed or taking too long."
    fi
fi

if [ -n "$DEBUG" ] && [ "$DEBUG" = "true" ]; then
    echo "🛠️ Debug mode enabled. Starting SSH server..."
    service ssh start
    
    echo "✅ SSH is ready. Container will stay alive."
    tail -f /dev/null # keep container alive for debugging
elif [ -n "$TRAIN_ARGS" ]; then
    # use env variables to build training args
    echo "Starting training with args: $TRAIN_ARGS"
    echo "--- Current Directory Content ---"
    ls -l /app | head -n 5
    git log -1 --oneline 2>/dev/null || echo "Not a git repo or sync skipped"
    uv run --no-sync train $TRAIN_ARGS
    # auto-destroy instance to save money
    if [ -n "$VAST_API_KEY" ] && [ -n "$CONTAINER_ID" ]; then
        echo "Task complete. Destroying instance $CONTAINER_ID to save money..."
        # sleep for a while to ensure all logs are flushed
        sleep 30 
        vastai destroy instance $CONTAINER_ID --api-key $VAST_API_KEY
    else
        echo "Task complete. Auto-destroy skipped (Missing API Key or ID)."
        # keep container alive for inspection
        # tail -f /dev/null
    fi
fi
