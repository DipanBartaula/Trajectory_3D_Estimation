import torch
import numpy as np
import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
import VRAM

def test_vae():
    print("Testing VAE (MichelangeloLikeAutoencoderWrapper)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Path relative to root
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(root_dir, "checkpoints/vae-088-0-bfloat16.ckpt")
    
    if not os.path.exists(ckpt_path):
        print(f"Error: VAE checkpoint not found at {ckpt_path}. Skipping test.")
        return

    try:
        vae = MichelangeloLikeAutoencoderWrapper(ckpt_path, device)
        vae.model = vae.model.to(dtype=torch.float16)
        vae.model.eval()
        
        embed_dim = vae.get_embed_dim()
        # Create dummy latents
        # ShapeR uses (1, token_count, embed_dim)
        # Token count depends on scale, usually 128 or 256
        token_count = 128 
        dummy_latents = torch.randn(1, token_count, embed_dim, device=device, dtype=torch.float16)
        
        print(f"Dummy Latents Shape: {dummy_latents.shape}")
        
        with torch.no_grad():
            # infer_mesh_from_latents returns a list of meshes
            meshes = vae.infer_mesh_from_latents(dummy_latents)
            
        print(f"Generated {len(meshes)} mesh(es).")
        if len(meshes) > 0:
            print(f"Mesh Vertices: {len(meshes[0].vertices)}")
            print(f"Mesh Faces: {len(meshes[0].faces)}")
            
        print("VAE Test Passed!")
    except Exception as e:
        print(f"VAE Test Failed: {e}")

if __name__ == "__main__":
    test_vae()
