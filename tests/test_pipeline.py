import argparse
import os
import torch
import pickle
import sys
from pathlib import Path

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer_shape import main as infer_main

def test_pipeline():
    parser = argparse.ArgumentParser(description="Test ShapeR Pipeline without re-running SfM")
    parser.add_argument("--input_pkl", type=str, help="Path to pre-generated .pkl file")
    parser.add_argument("--config", type=str, default="balance", help="Inference config")
    parser.add_argument("--output_dir", type=str, default="test_pipeline_output", help="Output directory")
    parser.add_argument("--no_text", action="store_true", help="Disable text conditioning (T5/CLIP)")
    
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if args.input_pkl is None:
        # Try to find any pkl in the root or data/
        pkl_files = list(Path(root_dir).glob("*.pkl")) + list(Path(root_dir).joinpath("data/processed").glob("*.pkl"))
        if pkl_files:
            args.input_pkl = str(pkl_files[0])
            print(f"Auto-selected PKL: {args.input_pkl}")
        else:
            print("Error: No .pkl file found. Please run run_inference.sh first or provide --input_pkl")
            return

    print(f"Testing pipeline with {args.input_pkl}...")
    
    # We can just invoke infer_shape.main with specific sys.argv
    sys.argv = [
        "infer_shape.py",
        "--input_pkl", args.input_pkl,
        "--config", args.config,
        "--output_dir", args.output_dir,
        "--save_visualization"
    ]
    if args.no_text:
        sys.argv.append("--no_text")
    
    try:
        infer_main()
        print("\nPipeline Test Completed Successfully!")
        print(f"Results saved to {args.output_dir}")
    except Exception as e:
        print(f"\nPipeline Test Failed: {e}")

if __name__ == "__main__":
    test_pipeline()
