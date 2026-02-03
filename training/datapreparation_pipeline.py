
import os
import glob
import subprocess
import pickle
import trimesh
import torch
import sys
from pathlib import Path
from tqdm import tqdm

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video_to_pkl import process_video

class DataPreparationPipeline:
    def __init__(self, input_dir, output_dir, device="cuda"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def run(self):
        video_extensions = ['*.mp4', '*.avi']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(self.input_dir.glob(ext)))
            
        print(f"Found {len(video_files)} videos.")
        
        for video_path in tqdm(video_files):
            video_name = video_path.stem
            output_pkl = self.output_dir / f"{video_name}.pkl"
            
            # Step 1: Video to PKL (Sparse Points + Images)
            if not output_pkl.exists():
                print(f"Processing {video_name} -> PKL")
                sam_checkpoint = "sam_vit_b_01ec64.pth" if os.path.exists("sam_vit_b_01ec64.pth") else "../sam_vit_b_01ec64.pth"
                process_video(str(video_path), str(output_pkl), sam_checkpoint=sam_checkpoint, device=self.device)
            
            # Step 2: Run Inference to get Pseudo-GT Mesh
            # We assume infer_shape.py is in project root
            # Output will be in 'output' dir by default or we specify
            mesh_output_dir = self.output_dir / "meshes"
            mesh_output_dir.mkdir(exist_ok=True)
            
            cmd = [
                "python", "infer_shape.py",
                "--input_pkl", str(output_pkl),
                "--output_dir", str(mesh_output_dir),
                "--config", "balance",
                "--save_visualization" # Optional
            ]
            
            glb_path = mesh_output_dir / f"{video_name}.glb"
            if not glb_path.exists():
                print(f"Running Inference for Pseudo-GT: {video_name}")
                # We need to run this from root dir so imports work
                subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
            # Step 3: Inject Mesh into PKL
            if glb_path.exists():
                print(f"Injecting Mesh into PKL: {video_name}")
                try:
                    # Load existing PKL
                    with open(output_pkl, "rb") as f:
                        data = pickle.load(f)
                    
                    # Load Mesh
                    mesh = trimesh.load(glb_path, force="mesh")
                    
                    # Update PKL
                    # Note: We might need to handle scaling/alignment if infer_shape rescaled it.
                    # Usually infer_shape outputs in world coordinates or similar.
                    # We store it as torch tensors
                    data["mesh_vertices"] = torch.tensor(mesh.vertices, dtype=torch.float32)
                    data["mesh_faces"] = torch.tensor(mesh.faces, dtype=torch.int32)
                    
                    # Save back
                    with open(output_pkl, "wb") as f:
                        pickle.dump(data, f)
                        
                except Exception as e:
                    print(f"Failed to inject mesh: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    
    pipeline = DataPreparationPipeline(args.input_dir, args.output_dir)
    pipeline.run()
