import argparse
import subprocess
import os
import shutil
import pickle
import numpy as np

def run_test(video_path):
    output_pkl = "test_output.pkl"
    
    # Clean up previous run
    if os.path.exists(output_pkl):
        os.remove(output_pkl)

    print(f"Testing video_to_pkl.py with video: {video_path}")
    
    # Run the script
    cmd = [
        "python", "video_to_pkl.py",
        "--video_path", video_path,
        "--output_pkl", output_pkl
    ]
    
    try:
        subprocess.check_call(cmd)
        print("\nSUCCESS: Execution completed without errors.")
    except subprocess.CalledProcessError as e:
        print(f"\nFAILURE: Script failed with exit code {e.returncode}")
        return

    # Verify output
    if not os.path.exists(output_pkl):
        print(f"\nFAILURE: Output file {output_pkl} was not created.")
        return

    print(f"\nVerifying {output_pkl} content...")
    try:
        with open(output_pkl, "rb") as f:
            data = pickle.load(f)
            
        required_keys = [
            "points_model", "bounds", "image_data", 
            "Ts_camera_model", "object_point_projections", 
            "camera_params", "caption", "visible_points_model"
        ]
        
        missing = [k for k in required_keys if k not in data]
        if missing:
             print(f"FAILURE: Missing keys in pickle: {missing}")
             return

        n_images = len(data["image_data"])
        n_poses = len(data["Ts_camera_model"])
        n_projs = len(data["object_point_projections"])
        
        print(f"Found {n_images} frames processed.")
        
        if not (n_images == n_poses == n_projs):
             print(f"FAILURE: Inconsistent list lengths: images={n_images}, poses={n_poses}, projs={n_projs}")
             return
             
        # Check torch tensors
        if not isinstance(data["points_model"], (np.ndarray, type(data["Ts_camera_model"][0]))):
             print("WARNING: points_model is not a tensor/array (check type).")

        print("SUCCESS: Pickle file seems valid!")
        print(f"Caption generated: {data['caption']}")
        print(f"Number of 3D points: {len(data['points_model'])}")

    except Exception as e:
        print(f"FAILURE: Failed to read pickle file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input test video")
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
    else:
        run_test(args.video_path)
