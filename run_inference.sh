#!/bin/bash

# Script to run ShapeR inference
# Usage: ./run_inference.sh [--video file.mp4 | --pkl file.pkl | --dataset name.pkl] [--device cuda]

DEVICE="cuda"
VIDEO_PATH=""
PKL_PATH=""
DATASET_PKL=""
DATASET_DIR="/mnt/shared_models/_home/pshrestha"
OUTPUT_DIR="output_shaper"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --video) VIDEO_PATH="$2"; shift ;;
        --pkl) PKL_PATH="$2"; shift ;;
        --dataset) DATASET_PKL="$2"; shift ;;
        --device) DEVICE="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -z "$VIDEO_PATH" ] && [ -z "$PKL_PATH" ] && [ -z "$DATASET_PKL" ]; then
    echo "Error: Must specify one of target inputs."
    echo "Usage: ./run_inference.sh [--video file.mp4 | --pkl file.pkl | --dataset name.pkl] [--device cuda]"
    exit 1
fi

echo "================================================="
echo "Config: balance"
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "================================================="

if [ -n "$VIDEO_PATH" ]; then
    VIDEO_NAME=$(basename "$VIDEO_PATH" .mp4)
    TARGET_PKL="$DATASET_DIR/$VIDEO_NAME.pkl"
    echo "Starting ShapeR Inference Pipeline (Video Mode)"
    echo "Video: $VIDEO_PATH"
    echo "Processed PKL will be saved to: $TARGET_PKL"
    echo "================================================="
    
    python infer_shape.py \
        --video_path "$VIDEO_PATH" \
        --input_pkl "$TARGET_PKL" \
        --config balance \
        --output_dir "$OUTPUT_DIR" \
        --dataset_dir "$DATASET_DIR" \
        --save_visualization \
        --remove_floating_geometry \
        --simplify_mesh

elif [ -n "$PKL_PATH" ]; then
    echo "Starting ShapeR Inference Pipeline (Local PKL Mode)"
    echo "Input PKL: $PKL_PATH"
    echo "================================================="
    
    python infer_shape.py \
        --input_pkl "$PKL_PATH" \
        --config balance \
        --output_dir "$OUTPUT_DIR" \
        --dataset_dir "$DATASET_DIR" \
        --save_visualization \
        --remove_floating_geometry \
        --simplify_mesh

elif [ -n "$DATASET_PKL" ]; then
    echo "Starting ShapeR Inference Pipeline (Dataset Evaluation Mode)"
    echo "Dataset PKL: $DATASET_PKL"
    echo "Will be loaded/downloaded from: $DATASET_DIR"
    echo "================================================="
    
    python infer_shape.py \
        --input_pkl "$DATASET_PKL" \
        --config balance \
        --output_dir "$OUTPUT_DIR" \
        --dataset_dir "$DATASET_DIR" \
        --save_visualization \
        --remove_floating_geometry \
        --simplify_mesh
fi

if [ $? -eq 0 ]; then
    echo "================================================="
    echo "Inference completed successfully!"
    echo "Check output in: $OUTPUT_DIR"
    echo "================================================="
else
    echo "================================================="
    echo "Inference Failed!"
    echo "================================================="
    exit 1
fi
