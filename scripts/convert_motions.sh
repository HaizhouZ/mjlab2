#!/bin/bash

# Script to convert NPZ motion files to CSV and then to NPZ format
# Usage: ./scripts/convert_motions.sh <motion-dir>

set -e  # Exit on error

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

# Loop through all subdirectories in motion-dir
for motion_path in "$MOTION_DIR"/*; do
    # Check if it's a directory
    if [ ! -d "$motion_path" ]; then
        continue
    fi
    
    # Get motion name (basename of the path)
    motion_name=$(basename "$motion_path")
    
    # Create subdirectory for this motion: motions/output/<motion-name>
    MOTION_OUTPUT_DIR="$OUTPUT_DIR/$motion_name"
    mkdir -p "$MOTION_OUTPUT_DIR"
    
    # CSV file path: motions/output/<motion-name>/motion.csv
    # The parent directory name (<motion-name>) will be used as collection name
    CSV_FILE="$MOTION_OUTPUT_DIR/motion.csv"
    
    # Path to best_trajectory.npz
    npz_file="$motion_path/best_trajectory.npz"
    
    # Check if best_trajectory.npz exists
    if [ ! -f "$npz_file" ]; then
        echo "Warning: '$npz_file' not found, skipping motion '$motion_name'"
        continue
    fi
    
    echo "=========================================="
    echo "Processing motion: $motion_name"
    echo "=========================================="
    
    # Step 1: Run new_conversion_1.py
    echo "Step 1: Converting NPZ to CSV..."
    uv run python -m mjlab.scripts.new_conversion_1 \
        --npz-file "$npz_file" \
        --csv-file "$CSV_FILE" \
        --add-start-transition \
        --add-end-transition \
        --transition-duration 1.5 \
        --pad-duration 0.5
    
    if [ $? -ne 0 ]; then
        echo "Error: Conversion 1 failed for motion '$motion_name'"
        continue
    fi
    
    # Step 2: Run new_conversion_2.py
    echo "Step 2: Converting CSV to NPZ..."
    uv run python -m mjlab.scripts.new_conversion_2 \
        --input-file "$CSV_FILE" \
        --project-name sbto_v1 \
        --render \
        --output-fps 50.0
    
    if [ $? -ne 0 ]; then
        echo "Error: Conversion 2 failed for motion '$motion_name'"
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

