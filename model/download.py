# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path

from huggingface_hub import snapshot_download


def setup_checkpoints(checkpoint_dir="./checkpoints"):
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    try:
        snapshot_download(
            repo_id="facebook/ShapeR",
            allow_patterns=["*.ckpt", "*.yaml"],
            local_dir=checkpoint_dir,
        )
    except Exception as e:
        print(f"Error downloading generic checkpoints: {e}")
        print("Note: If you don't have access to facebook/ShapeR, please place .ckpt and .yaml files in checkpoints/ manually.")
    
    # Download SAM Checkpoint specifically
    sam_ckpt = Path(checkpoint_dir) / "sam_vit_b_01ec64.pth"
    if not sam_ckpt.exists():
        print("Downloading SAM checkpoint...")
        import torch
        torch.hub.download_url_to_file(
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            str(sam_ckpt)
        )
