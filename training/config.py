
import os
import wandb
from huggingface_hub import login

# Credentials
# Ideally set these in your OS environment variables.
# Or replace the strings below with your actual keys if running locally (DO NOT COMMIT KEYS TO GIT).
WANDB_API_KEY = os.getenv("WANDB_API_KEY", "your_wandb_key_here")
HF_TOKEN = os.getenv("HF_TOKEN", "your_hf_token_here")

def login_services():
    """
    Log in to WandB and Hugging Face Hub using provided credentials.
    """
    # Login to WandB
    if WANDB_API_KEY and WANDB_API_KEY != "your_wandb_key_here":
        try:
            print(f"Logging into WandB (Key: ...{WANDB_API_KEY[-4:]})")
            wandb.login(key=WANDB_API_KEY)
        except Exception as e:
            print(f"Failed to login to WandB: {e}")
    else:
        print("WandB Key not found or default. Skipping auto-login (assuming manual login or local mode).")
    
    # Login to Hugging Face
    if HF_TOKEN and HF_TOKEN != "your_hf_token_here":
        try:
             print(f"Logging into Hugging Face (Token: ...{HF_TOKEN[-4:]})")
             login(token=HF_TOKEN)
        except Exception as e:
            print(f"Failed to login to HF: {e}")
    else:
        print("HF Token not found or default. Skipping auto-login.")
