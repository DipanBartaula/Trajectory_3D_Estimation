# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# pyre-unsafe

from enum import Enum
from typing import Union

import torch
import os

from .utils import _DINOV2_BASE_URL, _make_dinov2_model_name


class Weights(Enum):
    LVD142M = "LVD142M"


def _make_dinov2_model(
    *,
    arch_name: str = "vit_large",
    img_size: int = 518,
    patch_size: int = 14,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    num_register_tokens: int = 0,
    interpolate_antialias: bool = False,
    interpolate_offset: float = 0.1,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD142M,
    **kwargs,
):
    from ..models import vision_transformer as vits

    if isinstance(weights, str):
        try:
            weights = Weights[weights]
        except KeyError:
            raise AssertionError(f"Unsupported weights: {weights}")

    model_base_name = _make_dinov2_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        num_register_tokens=num_register_tokens,
        interpolate_antialias=interpolate_antialias,
        interpolate_offset=interpolate_offset,
    )
    vit_kwargs.update(**kwargs)
    model = vits.__dict__[arch_name](**vit_kwargs)

    if pretrained:
        model_full_name = _make_dinov2_model_name(
            arch_name, patch_size, num_register_tokens
        )
        # Check standard checkpoints location
        ckpt_filename = f"{model_full_name}_pretrain.pth"
        local_path = f"checkpoints/{ckpt_filename}"
        
        if os.path.exists(local_path):
             print(f"Loading DINOv2 weights from local file: {local_path}")
             state_dict = torch.load(local_path, map_location="cpu")
        elif "https://" in _DINOV2_BASE_URL or "http://" in _DINOV2_BASE_URL:
             # It's a URL, use hub load
             print(f"Downloading/Loading DINOv2 weights from URL: {_DINOV2_BASE_URL}/{ckpt_filename}")
             state_dict = torch.hub.load_state_dict_from_url(
                 _DINOV2_BASE_URL + f"/{ckpt_filename}", map_location="cpu"
             )
        else:
             # Assume it's a path but file is missing
             full_url_path = os.path.join(_DINOV2_BASE_URL, ckpt_filename)
             if os.path.exists(full_url_path):
                 state_dict = torch.load(full_url_path, map_location="cpu")
             else:
                 # Last resort: Try downloading from Hugging Face mirror as official Meta URL is 403 Forbidden
                 # Repo: facebook/dinov2-with-registers
                 fallback_url = f"https://huggingface.co/facebook/dinov2-with-registers/resolve/main/{ckpt_filename}"
                 print(f"Checkpoint not found locally. Downloading from {fallback_url}...")
                 
                 # Ensure checkpoints dir exists
                 os.makedirs("checkpoints", exist_ok=True)
                 try:
                     torch.hub.download_url_to_file(fallback_url, local_path)
                 except Exception as e:
                     print(f"Failed to download from Hugging Face: {e}")
                     # Try original one just in case, or another mirror
                     print("Trying alternate source...")
                     alt_url = f"https://dl.fbaipublicfiles.com/dinov2/{model_full_name}/{ckpt_filename}"
                     torch.hub.download_url_to_file(alt_url, local_path)

                 state_dict = torch.load(local_path, map_location="cpu")

        model.load_state_dict(state_dict, strict=True)

    return model


def dinov2_vits14(
    *, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs
):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_small", pretrained=pretrained, weights=weights, **kwargs
    )


def dinov2_vitb14(
    *, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs
):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_base", pretrained=pretrained, weights=weights, **kwargs
    )


def dinov2_vitl14(
    *, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs
):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_large", pretrained=pretrained, weights=weights, **kwargs
    )


def dinov2_vitg14(
    *, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs
):
    """
    DINOv2 ViT-g/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        **kwargs,
    )


def dinov2_vits14_reg(
    *, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs
):
    """
    DINOv2 ViT-S/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_small",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitb14_reg(
    *, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs
):
    """
    DINOv2 ViT-B/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_base",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitl14_reg(
    *, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs
):
    """
    DINOv2 ViT-L/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_large",
        pretrained=pretrained,
        weights=weights,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )


def dinov2_vitg14_reg(
    *, pretrained: bool = True, weights: Union[Weights, str] = Weights.LVD142M, **kwargs
):
    """
    DINOv2 ViT-g/14 model with registers (optionally) pretrained on the LVD-142M dataset.
    """
    return _make_dinov2_model(
        arch_name="vit_giant2",
        ffn_layer="swiglufused",
        weights=weights,
        pretrained=pretrained,
        num_register_tokens=4,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        **kwargs,
    )
