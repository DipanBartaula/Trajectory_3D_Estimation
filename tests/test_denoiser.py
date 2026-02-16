import torch
import omegaconf
import os
import sys

# Add root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.flow_matching.shaper_denoiser import ShapeRDenoiser

def test_denoiser():
    print("Testing ShapeRDenoiser...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(root_dir, "checkpoints/config.yaml")
    ckpt_path = os.path.join(root_dir, "checkpoints/019-0-bfloat16.ckpt")
    
    if not os.path.exists(config_path) or not os.path.exists(ckpt_path):
        print("Required checkpoints or config missing. Skipping Denoiser test.")
        return

    try:
        config = omegaconf.OmegaConf.load(config_path)
        model = ShapeRDenoiser(config).to(device)
        model.convert_to_fp16()
        
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        print("Denoiser loaded successfully.")
        
        # Create a dummy batch for testing
        token_count = 128
        embed_dim = 16 
        token_shape = (1, token_count, embed_dim)
        
        print(f"Model Training Target: {getattr(config.fm_transformer, 'time_sampler', 'lognorm')}")
        print("Denoiser Test (Loading) Passed!")
        
    except Exception as e:
        print(f"Denoiser Test Failed: {e}")

if __name__ == "__main__":
    test_denoiser()
