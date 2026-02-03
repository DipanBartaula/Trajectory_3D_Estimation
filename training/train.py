
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from omegaconf import OmegaConf

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import get_model, get_vae, get_text_encoder
from training.dataloader import get_dataloader
from training.utils import setup_wandb, log_metrics, save_checkpoint
from dataset.shaper_dataset import InferenceDataset

def flow_matching_loss(model, x_1, batch, device):
    """
    Compute Flow Matching Loss.
    """
    B, L, D = x_1.shape
    
    # x_0 is Gaussian noise (in FP16)
    x_0 = torch.randn_like(x_1).to(device)
    
    # Sample t
    t = torch.rand(B).to(device).to(x_1.dtype)
    
    # Reshape t
    t_expanded = t.view(B, 1, 1).expand(B, L, D)
    
    # Interpolate
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
    v_target = x_1 - x_0
    
    # Predict velocity
    v_pred = model(x_t, t, batch)
    
    loss = F.mse_loss(v_pred, v_target)
    return loss

def train_one_epoch(model, dataloader, optimizer, device, epoch, step_offset, dtype=torch.float16):
    model.train()
    total_loss = 0
    steps = step_offset
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move batch to device with correct dtype (FP16)
        batch = InferenceDataset.move_batch_to_device(batch, device, dtype=dtype)
        
        # Placeholder for GT latents (x_1)
        # Assuming we generate random target if not present, to allow code to run
        if "gt_latents" in batch:
             x_1 = batch["gt_latents"]
        else:
             # Dummy target in FP16
             x_1 = torch.randn(batch["images"].shape[0], 256, 128, device=device, dtype=dtype)

        optimizer.zero_grad()
        loss = flow_matching_loss(model, x_1, batch, device)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        
        log_metrics({"train/loss": loss.item(), "epoch": epoch}, steps)
        pbar.set_postfix(loss=loss.item())
        
        if steps % 250 == 0:
            save_checkpoint(model, optimizer, steps, "training/checkpoints")
            
    return steps

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = "checkpoints/config.yaml"
    ckpt_path = "checkpoints/019-0-bfloat16.ckpt"
    
    print("Initializing Training (FP16)...")
    model, config = get_model(config_path, ckpt_path, device, apply_lora=True, precision="fp16")
    
    dataloader = get_dataloader(config, "data", batch_size=2)
    if not dataloader:
        print("No data found, exiting.")
        return

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4)
    
    setup_wandb("shaper-lora-training", "run-001", OmegaConf.to_container(config))
    
    global_step = 0
    start_epoch = 0
    
    # Resume Logic
    checkpoint_dir = "training/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if checkpoints:
        checkpoints.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_dir, x)))
        latest_ckpt = os.path.join(checkpoint_dir, checkpoints[-1])
        print(f"Resuming from {latest_ckpt}")
        
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(ckpt_data['model_state_dict'], strict=False)
        optimizer.load_state_dict(ckpt_data['optimizer_state_dict'])
        global_step = ckpt_data.get('step', 0)
        print(f"Resumed at step {global_step}")
    
    epochs = 100
    for epoch in range(start_epoch, epochs):
        global_step = train_one_epoch(model, dataloader, optimizer, device, epoch, global_step, dtype=torch.float16)

if __name__ == "__main__":
    main()
