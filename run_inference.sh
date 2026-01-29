#!/bin/bash

# Script to run ShapeR inference on a video file
# Usage: ./run_inference.sh <path_to_video.mp4> [device]

VIDEO_PATH=$1
DEVICE=${2:-cuda}  # Default to cuda if not specified

if [ -z "$VIDEO_PATH" ]; then
  echo "Error: No video path provided."
  echo "Usage: ./run_inference.sh <path_to_video.mp4> [device]"
  echo "Example: ./run_inference.sh my_video.mp4"
  exit 1
fi

OUTPUT_DIR="output_shaper"

echo "================================================="
echo "Starting ShapeR Inference Pipeline"
echo "Video: $VIDEO_PATH"
echo "Config: balance"
echo "Output Directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "================================================="

# Ensure conda environment is active (optional check)
# if [[ "$CONDA_DEFAULT_ENV" != "shaper" ]]; then
#     echo "Warning: Conda environment 'shaper' is not active."
#     echo "You might want to run: conda activate shaper"
# fi

python infer_shape.py \
    --video_path "$VIDEO_PATH" \
    --config balance \
    --output_dir "$OUTPUT_DIR" \
    --save_visualization \
    --remove_floating_geometry \
    --simplify_mesh

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
