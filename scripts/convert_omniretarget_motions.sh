#!/bin/bash

# Script to convert NPZ motion files to CSV and then to NPZ format
# Usage: ./scripts/convert_motions.sh <motion-dir>


# Check if motion-dir argument is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <motion-dir>"
    echo "Example: $0 motions/input"
    exit 1
fi

MOTION_DIR="$1"

# Check if motion-dir exists
if [ ! -d "$MOTION_DIR" ]; then
    echo "Error: Directory '$MOTION_DIR' does not exist"
    exit 1
fi

# Create output directory if it doesn't exist
OUTPUT_DIR="motions/output"
mkdir -p "$OUTPUT_DIR"

# SBTO directory path
SBTO_DIR="../../SBTO_OmniRetarget_Dataset"

# Array to track failed motions
FAILED_MOTIONS=()

# Loop through all .npz files in motion-dir
for npz_file in "$MOTION_DIR"/*.npz; do
    # Check if any .npz files exist (glob might not match anything)
    if [ ! -f "$npz_file" ]; then
        echo "Warning: No .npz files found in '$MOTION_DIR'"
        break
    fi
    
    # Get motion name (filename without .npz extension)
    motion_name=$(basename "$npz_file" .npz)
    
    # Check if this motion exists in SBTO directory
    sbto_motion_file="$SBTO_DIR/$motion_name/best_trajectory.npz"
    if [ ! -f "$sbto_motion_file" ]; then
        echo "Skipping motion '$motion_name' (not found in SBTO directory: $sbto_motion_file)"
        continue
    fi
    
    # Create subdirectory for this motion: motions/output/<motion-name>
    MOTION_OUTPUT_DIR="$OUTPUT_DIR/$motion_name"
    mkdir -p "$MOTION_OUTPUT_DIR"
    
    # CSV file path: motions/output/<motion-name>/motion.csv
    # The parent directory name (<motion-name>) will be used as collection name
    CSV_FILE="$MOTION_OUTPUT_DIR/motion.csv"
    
    echo "=========================================="
    echo "Processing motion: $motion_name"
    echo "=========================================="
    
    # Step 1: Run new_conversion_1.py
    echo "Step 1: Converting NPZ to CSV..."
    if ! uv run python -m mjlab.scripts.new_conversion_1 \
        --input-file "$npz_file" \
        --csv-file "$CSV_FILE" \
        --add-start-transition \
        --add-end-transition \
        --transition-duration 1.5 \
        --pad-duration 0.5; then
        echo "Error: Conversion 1 failed for motion '$motion_name'"
        FAILED_MOTIONS+=("$motion_name")
        continue
    fi
    
    # Step 2: Run new_conversion_2.py
    echo "Step 2: Converting CSV to NPZ..."
    if ! uv run python -m mjlab.scripts.new_conversion_2 \
        --input-file "$CSV_FILE" \
        --task-name Mjlab-Tracking-Flat-Unitree-G1-LargeBox-No-State-Estimation \
        --project-name omniretarget-v1-motions \
        --render \
        --output-fps 50.0 --input-fps 30.0; then
        echo "Error: Conversion 2 failed for motion '$motion_name'"
        FAILED_MOTIONS+=("$motion_name")
        continue
    fi
    
    # Delete CSV file and empty directory after processing
    rm -f "$CSV_FILE"
    rmdir "$MOTION_OUTPUT_DIR" 2>/dev/null || true
    
    echo "Successfully processed motion: $motion_name"
    echo ""
done

echo "=========================================="
echo "All motions processed!"
echo "=========================================="

# Print failed motions if any
if [ ${#FAILED_MOTIONS[@]} -gt 0 ]; then
    echo ""
    echo "=========================================="
    echo "Failed motions (${#FAILED_MOTIONS[@]} total):"
    echo "=========================================="
    for motion in "${FAILED_MOTIONS[@]}"; do
        echo "  - $motion"
    done
    echo "=========================================="
else
    echo ""
    echo "All motions processed successfully!"
fi

