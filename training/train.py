
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from omegaconf import OmegaConf
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import get_model, get_vae, get_text_encoder
from training.dataloader import get_dataloader
from training.utils import log_metrics, save_checkpoint # save_checkpoint might be retired for accelerator.save_state
from training.config import login_services
from dataset.shaper_dataset import InferenceDataset

def flow_matching_loss(model, x_1, batch, device):
    """
    Compute Flow Matching Loss.
    """
    B, L, D = x_1.shape
    
    # x_0 is Gaussian noise
    x_0 = torch.randn_like(x_1) # device handled by accelerator if x_1 on device
    
    # Sample t
    t = torch.rand(B, device=device).to(x_1.dtype)
    
    # Reshape t
    t_expanded = t.view(B, 1, 1).expand(B, L, D)
    
    # Interpolate
    x_t = t_expanded * x_1 + (1 - t_expanded) * x_0
    v_target = x_1 - x_0
    
    # Predict velocity
    v_pred = model(x_t, t, batch)
    
    loss = F.mse_loss(v_pred, v_target)
    return loss

def train_one_epoch(model, dataloader, optimizer, accelerator, epoch, step_offset):
    model.train()
    total_loss = 0
    steps = step_offset
    
    # Disable tqdm on non-main processes to reduce log noise
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not accelerator.is_local_main_process)
    
    for batch in pbar:
        # Move complex types (SparseTensor) manually if Accelerator misses them
        # Accelerator handles standard Tensors in dicts usually
        batch = InferenceDataset.move_batch_to_device(batch, accelerator.device)
        
        # Placeholder for GT latents
        if "gt_latents" in batch:
             x_1 = batch["gt_latents"]
        else:
             # Dummy target
             x_1 = torch.randn(batch["images"].shape[0], 256, 128, device=accelerator.device, dtype=torch.float32)

        # Optimization Step
        # No zero_grad needed if set_to_none=True in init (default)
        optimizer.zero_grad()
        
        loss = flow_matching_loss(model, x_1, batch, accelerator.device)
        
        accelerator.backward(loss)
        optimizer.step()
        
        total_loss += loss.item()
        steps += 1
        
        # Logging
        if accelerator.is_main_process:
             accelerator.log({"train/loss": loss.item(), "epoch": epoch, "step": steps}, step=steps)
             pbar.set_postfix(loss=loss.item())
        
        # Save Checkpoint
        if steps % 250 == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_path = os.path.join("training/checkpoints", f"checkpoint_{steps}")
                accelerator.save_state(save_path)
            
    return steps

def main():
    login_services()
    
    # Initialize Accelerator
    project_config = ProjectConfiguration(project_dir=".", logging_dir="logs")
    accelerator = Accelerator(
        mixed_precision="fp16",
        log_with="wandb",
        project_config=project_config
    )
    
    if accelerator.is_main_process:
        print("Initializing Training with Accelerator...")
    
    # Configuration
    config_path = "checkpoints/config.yaml"
    ckpt_path = "checkpoints/019-0-bfloat16.ckpt"
    
    # Load Model (Base in FP16/BF16, LoRA in FP32 usually)
    # We pass precision="fp16" to convert Base model.
    model, config = get_model(config_path, ckpt_path, accelerator.device, apply_lora=True, precision="fp16")
    
    # Data
    dataloader = get_dataloader(config, "data", batch_size=2)
    if not dataloader: 
        if accelerator.is_main_process: print("No data found."); return

    # Optimizer (Only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4)

    # Prepare with Accelerator
    # Note: model, optimizer, dataloader are prepared.
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Init Logging
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="shaper-lora-training", 
            config=OmegaConf.to_container(config),
            init_kwargs={"wandb": {"name": "accelerate-run"}}
        )

    # Resume Logic
    checkpoint_dir = "training/checkpoints"
    global_step = 0
    start_epoch = 0
    # Search for accelerator checkpoints (directories) like checkpoint_1000
    # Logic: Look for folders starting with checkpoint_
    if os.path.exists(checkpoint_dir):
        subdirs = [d for d in os.listdir(checkpoint_dir) if d.startswith("checkpoint_")]
        if subdirs:
            # Sort by integer step
            subdirs.sort(key=lambda x: int(x.split('_')[1]))
            latest_ckpt = os.path.join(checkpoint_dir, subdirs[-1])
            if accelerator.is_main_process: print(f"Resuming from {latest_ckpt}")
            accelerator.load_state(latest_ckpt)
            try:
                global_step = int(subdirs[-1].split('_')[1])
            except: pass

    epochs = 100
    for epoch in range(start_epoch, epochs):
        global_step = train_one_epoch(model, dataloader, optimizer, accelerator, epoch, global_step)
        
    accelerator.end_training()

if __name__ == "__main__":
    main()
