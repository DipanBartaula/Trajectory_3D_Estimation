
import os
import sys
import torch
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import get_model
from dataset.shaper_dataset import InferenceDataset

def predict(video_pkl, checkpoint_path, lora_weights_path=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Base in FP16
    model, config = get_model("checkpoints/config.yaml", checkpoint_path, device, apply_lora=True, precision="fp16")
    
    # Load LoRA
    if lora_weights_path:
        print(f"Loading LoRA from {lora_weights_path}")
        checkpoint = torch.load(lora_weights_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
    model.eval()
    print("Model loaded in FP16 for inference.")
    
    # Logic similar to infer_shape can be added here
    # Since model is now FP16, ensure inputs are FP16
    
if __name__ == "__main__":
    predict("sample.pkl", "checkpoints/019-0-bfloat16.ckpt")
