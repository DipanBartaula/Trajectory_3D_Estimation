
import numpy as np
import torch
import torchsparse
import pickle
import cv2
import os
import argparse
import subprocess
from PIL import Image
import io

def create_dumb_pickle(output_path="dummy_inference.pkl"):
    print("Creating dummy pickle file for testing...")
    
    num_points = 500
    num_views = 3
    image_size = (255, 255) # H, W matches helper logic somewhat
    
    # 1. Random Point Cloud
    points_model = torch.randn(num_points, 3, dtype=torch.float32)
    # Center points slightly
    points_model = points_model - points_model.mean(dim=0)
    print(f"Points Model Shape: {points_model.shape}")
    
    # 2. Random Images (Bytes)
    pkl_image_data = []
    print(f"Generating {num_views} random images of size {image_size}...")
    for _ in range(num_views):
        # Create random image
        img_np = np.random.randint(0, 255, (image_size[0], image_size[1], 3), dtype=np.uint8)
        # Encode to bytes
        success, encoded_img = cv2.imencode('.png', img_np)
        if success:
             pkl_image_data.append(encoded_img.tobytes())
    
    # 3. Random Camera Params (16 elements for Fisheye624 compatibility)
    # [fx, fy, cx, cy, k1..k6, p1, p2, s1..s4]
    pkl_camera_params = []
    for _ in range(num_views):
        params = torch.zeros(16, dtype=torch.float32)
        params[0] = 300.0 # fx
        params[1] = 300.0 # fy
        params[2] = image_size[1] / 2.0 # cx
        params[3] = image_size[0] / 2.0 # cy
        pkl_camera_params.append(params)
    print(f"Camera Params Shape (per view): {pkl_camera_params[0].shape}")

    # 4. Extrinsics (Ts_camera_model) - World to Camera
    # Just use identity or simple lookat
    pkl_Ts_camera_model = []
    for _ in range(num_views):
        T = torch.eye(4, dtype=torch.float32)
        # Add some random translation to avoid 0
        T[:3, 3] = torch.randn(3)
        pkl_Ts_camera_model.append(T)
    print(f"Extrinsics Shape (per view): {pkl_Ts_camera_model[0].shape}")

    # 5. Projections & Visibility (Randomly select points)
    pkl_object_point_projections = []
    pkl_visible_points_model = []
    
    for _ in range(num_views):
        # Select random subset of points visible
        num_visible = np.random.randint(10, 50)
        visible_indices = np.random.choice(num_points, num_visible, replace=False)
        pkl_visible_points_model.append(visible_indices)
        
        # Fake projections (UV)
        projections = torch.rand(num_visible, 2, dtype=torch.float32) * image_size[0]
        pkl_object_point_projections.append(projections)

    print(f"Visible Points (view 0): {pkl_visible_points_model[0].shape}, Projections: {pkl_object_point_projections[0].shape}")

    data = {
        "points_model": points_model,
        "bounds": torch.tensor(1.0, dtype=torch.float32),
        "inv_dist_std": torch.zeros(num_points, dtype=torch.float32),
        "dist_std": torch.zeros(num_points, dtype=torch.float32),
        "image_data": pkl_image_data,
        "Ts_camera_model": pkl_Ts_camera_model,
        "object_point_projections": pkl_object_point_projections,
        "camera_params": pkl_camera_params,
        "caption": "a 3D object test",
        "category": "object",
        "T_model_world": torch.eye(4, dtype=torch.float32),
        "T_zup_obj": torch.eye(4, dtype=torch.float32),
        "visible_points_model": pkl_visible_points_model
    }
    
    print(f"Saving dummy pickle to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    print("Done.")

def run_inference(pkl_path):
    print("\nRunning infer_shape.py with dummy data...")
    cmd = [
        "python", "infer_shape.py",
        "--input_pkl", pkl_path,
        "--output_dir", "test_output",
        "--config", "m18",
        "--save_mesh"
    ]
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    pkl_name = "test_dummy.pkl"
    create_dumb_pickle(pkl_name)
    run_inference(pkl_name)
