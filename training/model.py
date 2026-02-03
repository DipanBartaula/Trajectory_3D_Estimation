
import os
import sys
import math
import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.flow_matching.dualstream_transformer import DoubleStreamBlock, SingleStreamBlock, SelfAttention
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
from model.text.hf_embedder import TextFeatureExtractor

class LoRALinear(nn.Module):
    def __init__(self, original_linear, r=8, alpha=16, dropout=0.1):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.original_linear = original_linear
        # freeze original
        for p in self.original_linear.parameters():
            p.requires_grad = False
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        
        # Init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x):
        return self.original_linear(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

def inject_lora(model, r=8, alpha=16):
    """
    Recursively find linear layers in target modules and replace them with LoRA layers.
    Simple heuristic: Iterate over transformer blocks.
    """
    if hasattr(model, 'transformer'):
        tf = model.transformer
        
        # Double Stream Blocks
        if hasattr(tf, 'double_blocks'):
            for block in tf.double_blocks:
                replace_linear_in_block(block, r, alpha)
        
        # Text Blocks
        if hasattr(tf, 'text_blocks'):
            for block in tf.text_blocks:
                replace_linear_in_block(block, r, alpha)
                
        # Single Stream Blocks
        if hasattr(tf, 'single_blocks'):
            for block in tf.single_blocks:
                replace_linear_in_block(block, r, alpha)

def replace_linear_in_block(block, r, alpha):
    """
    Replace Linear layers within a block with LoRALinear.
    """
    for name, module in block.named_children():
        if isinstance(module, nn.Linear):
            # Check if likely QKV/Proj or MLP (dimensions usually large)
            # We wrap all large linears
            input_dim = module.in_features
            if input_dim > 128: 
                setattr(block, name, LoRALinear(module, r=r, alpha=alpha))
        elif isinstance(module, (SelfAttention, nn.Sequential)): 
             replace_linear_in_block(module, r, alpha)

def get_model(config_path, checkpoint_path, device, apply_lora=True, precision="fp16"):
    """
    Load the ShapeR model.
    """
    config = OmegaConf.load(config_path)
    model = ShapeRDenoiser(config).to(device)
    
    # Load weights
    print(f"Loading weights from {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    
    # Set Precision
    if precision == "fp16":
        print("Converting model to float16")
        model.convert_to_fp16()
    elif precision == "bf16":
        print("Converting model to bfloat16")
        model.convert_to_bfloat16()
    
    # Freeze model initially
    for p in model.parameters():
        p.requires_grad = False
        
    if apply_lora:
        print("Applying LoRA...")
        inject_lora(model, r=16, alpha=32)
        
        # Ensure only LoRA parameters are trainable
        trainable_params = 0
        all_params = 0
        for p in model.parameters():
            all_params += p.numel()
            if p.requires_grad:
                trainable_params += p.numel()
        print(f"Trainable Parameters: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")
    
    return model, config

def get_vae(ckpt_path, device):
    return MichelangeloLikeAutoencoderWrapper(ckpt_path, device)

def get_text_encoder(device, dtype=torch.float16):
    tex = TextFeatureExtractor(device=device)
    tex = tex.to(dtype)
    return tex
