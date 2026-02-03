
import os
import torch
import wandb
import numpy as np

def setup_wandb(project_name, run_name, config):
    try:
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            reinit=True
        )
    except Exception as e:
        print(f"WandB setup failed (ignoring): {e}")

def log_metrics(metrics, step):
    try:
        wandb.log(metrics, step=step)
    except:
        pass

def save_checkpoint(model, optimizer, step, save_dir, filename=None):
    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"checkpoint_{step}.pth"
    save_path = os.path.join(save_dir, filename)
    
    # Save only trainable params (LoRA)
    trainable_state_dict = {k: v for k, v in model.state_dict().items() if v.requires_grad}
    
    torch.save({
        'step': step,
        'model_state_dict': trainable_state_dict, 
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    print(f"Saved checkpoint to {save_path}")
