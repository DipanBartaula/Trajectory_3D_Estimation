
import torch
import gc

def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb

def estimate_required_vram(fp16=True):
    """
    Estimate VRAM requirements for ShapeR inference components in GB.
    Based on typical parameter counts for these architectures.
    """
    bytes_per_param = 2 if fp16 else 4
    
    # Approximate Parameter Counts (Billions)
    models = {
        "T5_XL": 2.85,      # ~3B
        "DINO_Giant": 1.1,  # ~1.1B
        "ShapeR_Trans": 0.6, # ~600M (Estimate)
        "VAE": 0.1,         # ~100M
        "CLIP": 0.4,        # ~400M
        "SAM": 0.1          # ~100M (ViT-B)
    }
    
    # Base Weights Memory
    total_weights_gb = sum(models.values()) * 1e9 * bytes_per_param / 1024**3
    
    # Activation Overhead Estimate (Batch=1)
    # DINO and T5 attentions are memory hungry
    activation_overhead_gb = 4.0 
    
    total_required = total_weights_gb + activation_overhead_gb
    return total_required, models

def get_available_vram(device_id=0):
    """Get available VRAM on the specific device in GB."""
    if not torch.cuda.is_available():
        return 0
    
    # total_mem = torch.cuda.get_device_properties(device_id).total_memory
    # allocated = torch.cuda.memory_allocated(device_id)
    # reserved = torch.cuda.memory_reserved(device_id)
    # free = total_mem - reserved 
    
    # Use mem_get_info for free/total
    free_mem, total_mem = torch.cuda.mem_get_info(device_id)
    
    return free_mem / 1024**3

def should_offload(device_id=0, safety_margin_gb=2.0):
    """
    Decide if we should run in 'sequential/offload' mode.
    """
    required, _ = estimate_required_vram()
    available = get_available_vram(device_id)
    
    print(f"[VRAM Check] Required: ~{required:.2f} GB | Available: {available:.2f} GB")
    
    if available < (required + safety_margin_gb):
        print(">> Low VRAM detected. Enabled sequential model loading (Offloading).")
        return True
    else:
        print(">> Sufficient VRAM. Loading all models.")
        return False

def move_to_device(model, device):
    model.to(device)
    torch.cuda.empty_cache()

def offload_to_cpu(model):
    model.to("cpu")
    torch.cuda.empty_cache()
    gc.collect()

def print_vram_usage(stage=""):
    """Prints current VRAM usage."""
    if not torch.cuda.is_available():
        return
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    free_mem, total_mem = torch.cuda.mem_get_info()
    free_gb = free_mem / 1024**3
    total_gb = total_mem / 1024**3
    
    print(f"[{stage}] VRAM: Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={free_gb:.2f}GB / {total_gb:.2f}GB")

