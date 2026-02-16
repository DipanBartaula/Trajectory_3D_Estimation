#!/bin/bash

# Script to run ShapeR inference WITHOUT text conditioning (no T5/CLIP loading)
# Usage: bash bash/run_inference_text.sh <path_to_video.mp4> [device]

# Change to project root
cd "$(dirname "$0")/.."

VIDEO_PATH=$1
DEVICE=${2:-cuda}  # Default to cuda if not specified

if [ -z "$VIDEO_PATH" ]; then
  echo "Error: No video path provided."
  echo "Usage: bash bash/run_inference_text.sh <path_to_video.mp4> [device]"
  exit 1
fi

OUTPUT_DIR="output_shaper_no_text"
PROCESSED_DIR="data/processed"
mkdir -p "$PROCESSED_DIR"

VIDEO_NAME=$(basename "$VIDEO_PATH" .mp4)
PKL_PATH="$PROCESSED_DIR/${VIDEO_NAME}.pkl"

echo "================================================="
echo "Starting ShapeR Inference Pipeline (NO TEXT MODE)"
echo "Video: $VIDEO_PATH"
echo "Processed PKL will be saved to: $PKL_PATH"
echo "Config: balance"
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Note: Text conditioning and T5 model are DISABLED."
echo "================================================="

python infer_shape.py \
    --video_path "$VIDEO_PATH" \
    --input_pkl "$PKL_PATH" \
    --config balance \
    --output_dir "$OUTPUT_DIR" \
    --no_text \
    --save_visualization \
    --remove_floating_geometry \
    --simplify_mesh

if [ $? -eq 0 ]; then
    echo "================================================="
    echo "Inference completed successfully (No Text)!"
    echo "Check output in: $OUTPUT_DIR"
    echo "================================================="
else
    echo "================================================="
    echo "Inference Failed!"
    echo "================================================="
    exit 1
fi
