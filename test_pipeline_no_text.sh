#!/bin/bash

# Fast testing script that uses ALREADY GENERATED SfM points (.pkl)
# and skips text/T5 conditioning for maximum speed and lower VRAM.
# Usage: ./test_pipeline_no_text.sh <path_to_pkl_file>

PKL_PATH=$1

if [ -z "$PKL_PATH" ]; then
  # Try to find the most recent processed PKL
  LATEST_PKL=$(ls -t data/processed/*.pkl 2>/dev/null | head -n 1)
  if [ -z "$LATEST_PKL" ]; then
    echo "Error: No .pkl file provided and none found in data/processed/"
    echo "Usage: ./test_pipeline_no_text.sh <path_to_pkl_file>"
    exit 1
  fi
  PKL_PATH=$LATEST_PKL
  echo "Auto-selected latest processed PKL: $PKL_PATH"
fi

OUTPUT_DIR="test_pipeline_no_text_output"

echo "================================================="
echo "Testing ShapeR Pipeline (FAST MODE: PKL + NO TEXT)"
echo "Input PKL: $PKL_PATH"
echo "Output Directory: $OUTPUT_DIR"
echo "================================================="

python test_pipeline.py \
    --input_pkl "$PKL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --no_text

if [ $? -eq 0 ]; then
    echo "Test completed successfully!"
else
    echo "Test failed!"
    exit 1
fi
