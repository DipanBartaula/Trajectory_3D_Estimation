
import os
import sys
import torch
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.shaper_dataset import InferenceDataset

def get_dataloader(config, data_dir, batch_size=4, num_workers=4, shuffle=True):
    """
    Creates a DataLoader for training.
    """
    pkl_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pkl')]
    
    if not pkl_files:
        if os.path.exists("sample.pkl"): # Fallback for demo
             pkl_files = ["sample.pkl"]
        else:
             print(f"Warning: No .pkl files found in {data_dir}")
             return []
        
    print(f"Creating dataset with {len(pkl_files)} samples.")
    
    dataset = InferenceDataset(
        config,
        paths=pkl_files,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True if len(dataset) > batch_size else False,
        num_workers=num_workers,
        collate_fn=dataset.custom_collate,
        pin_memory=True
    )
    
    return dataloader
