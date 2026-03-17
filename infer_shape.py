# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.

"""
ShapeR Inference Script

Reconstructs 3D meshes from SLAM observations (point clouds + images + text).
Uses flow matching to generate latent codes, then decodes them via a 3D VAE.

Usage:
    python infer_shape.py --input_pkl <sample.pkl> --config balance --save_visualization
"""

import argparse
import os
# Increase timeout for Hugging Face downloads to prevent ReadTimeout
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from pathlib import Path

import numpy as np
import omegaconf
import torch

# important! We are using an old version of torchsparse, please use the legacy version otherwise you will get errors,\
# since torchsparse changed their datastructures in newer versions

import trimesh
from dataset.download import setup_data
from dataset.shaper_dataset import InferenceDataset
from model.download import setup_checkpoints
from model.flow_matching.shaper_denoiser import ShapeRDenoiser
from model.text.hf_embedder import TextFeatureExtractor
from model.vae3d.autoencoder import MichelangeloLikeAutoencoderWrapper
from postprocessing.helper import (
    remove_floating_geometry,
    visualize_prediction_and_groundtruth,
)
from tqdm import tqdm

import sys
# Add current directory to path to support importing video_to_pkl
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from video_to_pkl import process_video
import VRAM
import gc

# @lint-ignore-every PYTHONPICKLEISBAD

# Preset configs: (num_images, token_multiplier, num_denoising_steps)
# quality: Best results, slowest inference
# speed: Fastest inference, lower quality
# balance: Good tradeoff between quality and speed
preset_configs = {
    "quality": (16, 4, 50),
    "speed": (4, 2, 10),
    "balance": (16, 4, 25),
}


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to the input video file. If provided, input_pkl will be generated from this video.",
    )
    parser.add_argument(
        "--input_pkl",
        type=str,
        default=None,
        help="Path to the input pkl file. If video_path is provided, this is ignored (or used as output path for generated pkl).",
    )
    parser.add_argument(
        "--remove_floating_geometry",
        action="store_false",
        help="Remove floating geometry from the mesh.",
    )
    parser.add_argument(
        "--simplify_mesh",
        action="store_false",
        help="Simplify the mesh.",
    )
    parser.add_argument(
        "--save_visualization",
        action="store_true",
        help="Visualize the input, output and ground truth.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Path to the output mesh.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="balance",
        help="Config to use for the inference.",
    )
    parser.add_argument(
        "--do_transform_to_world",
        action="store_true",
        help="Transform the mesh to world coordinates.",
    )
    parser.add_argument(
        "--no_text",
        action="store_true",
        help="Disable text conditioning and T5/CLIP model loading.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/mnt/shared_models/_home/pshrestha",
        help="Path where checkpoints are stored.",
    )
    parser.add_argument(
        "--force_reprocess",
        action="store_true",
        help="Force reprocessing the video even if the output PKL already exists.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="data",
        help="Path where dataset is/will be downloaded and evaluated from.",
    )

    args = parser.parse_args()

    # Handle video input
    if args.video_path:
        if not os.path.exists(args.video_path):
            raise FileNotFoundError(f"Video file not found: {args.video_path}")
        
        if args.input_pkl is None:
            # Generate a default pkl name based on video name
            video_name = Path(args.video_path).stem
            args.input_pkl = f"{video_name}.pkl"
            
        pkl_exists = False
        if os.path.exists(args.input_pkl):
            pkl_exists = True
        else:
            possible_path = os.path.join(args.dataset_dir, args.input_pkl)
            if os.path.exists(possible_path):
                pkl_exists = True
                args.input_pkl = possible_path

        if pkl_exists and not args.force_reprocess:
            print(f"PKL file {args.input_pkl} already exists. Skipping video preprocessing. Use --force_reprocess to overwrite.")
        else:
            print(f"Processing video {args.video_path} -> {args.input_pkl}")
            
            # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Check for SAM checkpoint
        sam_ckpt_dir = args.checkpoint_dir
        if not os.path.exists(sam_ckpt_dir):
            os.makedirs(sam_ckpt_dir, exist_ok=True)
            
        sam_ckpt = os.path.join(sam_ckpt_dir, "sam_vit_b_01ec64.pth")
        
        if not os.path.exists(sam_ckpt):
            # Check root fallback
            if os.path.exists("sam_vit_b_01ec64.pth"):
                sam_ckpt = "sam_vit_b_01ec64.pth"
            else:
                # Download it!
                print(f"SAM checkpoint not found. Downloading to {sam_ckpt}...")
                sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                try:
                    torch.hub.download_url_to_file(sam_url, sam_ckpt)
                    print("Download complete.")
                except Exception as e:
                    print(f"Failed to download SAM checkpoint: {e}")
                    sam_ckpt = None # Logic downstream needs to handle this or crash
            
            process_video(args.video_path, args.input_pkl, sam_checkpoint=sam_ckpt, device=device)
    
    if not args.input_pkl:
        # Fallback default if nothing provided (though arguments usually handle defaults, we changed default to None)
        args.input_pkl = "ADT1292__stool.pkl"
        print(f"No input provided, using default: {args.input_pkl}")
    
    # Check if input_pkl exists in dataset_dir if not found locally
    if not os.path.exists(args.input_pkl) and not args.video_path:
        possible_path = os.path.join(args.dataset_dir, args.input_pkl)
        if os.path.exists(possible_path):
            args.input_pkl = possible_path
    
    print(f"Inference Config: {args.config} (Views: {preset_configs[args.config][0]}, Tokens*:{preset_configs[args.config][1]}, Steps: {preset_configs[args.config][2]})")

    num_images, token_multiplier, num_steps = preset_configs[args.config]

    # Download or verify checkpoints
    setup_checkpoints(checkpoint_dir=args.checkpoint_dir)
    
    # Only setup data if we need to download it.
    # If we generated it from video or it exists, we don't need to download it.
    if not os.path.exists(args.input_pkl) and not args.video_path:
        # Pass the relative path directly so directory structure is preserved during download
        setup_data(args.input_pkl, download_dir=args.dataset_dir)
        args.input_pkl = os.path.join(args.dataset_dir, args.input_pkl)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # load the checkpoint
    ckpt_file = os.path.join(args.checkpoint_dir, "019-0-bfloat16.ckpt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(ckpt_file, map_location=device, weights_only=False)

    # load the config (usually located in the folder above checkpoint)
    yaml_file = os.path.join(args.checkpoint_dir, "config.yaml")
    config = omegaconf.OmegaConf.load(yaml_file)
    # load the model and weights
    print("Loading model...")
    model = ShapeRDenoiser(config).to(device)
    model.convert_to_fp16()
    model.load_state_dict(state_dict, strict=False)
    print("Model loaded successfully.")

    vae = MichelangeloLikeAutoencoderWrapper(
        os.path.join(args.checkpoint_dir, "vae-088-0-bfloat16.ckpt"), device
    )
    val = vae.model.to(dtype=torch.float16)
    
    print("Model loaded successfully.")
    VRAM.print_vram_usage("After Model Load")

    if not args.no_text:
        text_feature_extractor = TextFeatureExtractor(device=device)
        text_feature_extractor = text_feature_extractor.to(torch.float16)
    else:
        from model.text.hf_embedder import DummyTextFeatureExtractor
        text_feature_extractor = DummyTextFeatureExtractor(device=device)

    # model = torch.compile(model, fullgraph=True) # Disable for stability with torchsparse/offloading
    model = model.eval()
    vae.model.use_udf_extraction = True
    vae.model.udf_iso = 0.375

    scales = vae.model.get_token_scales()
    scale_prob = np.zeros_like(scales)
    scale_prob[6] = 1.0
    vae.model.set_inference_scale_probabilities(scale_prob)
    token_count = int(scales[np.argmax(scale_prob)].item()) * token_multiplier
    token_shape = (1, token_count, vae.get_embed_dim())
    use_shifted_sampling = (
        getattr(config.fm_transformer, "time_sampler", "lognorm") == "flux"
    )

    # create batch sample
    print("Loading input pkl from", args.input_pkl)
    
    # Check if the path exists directly, otherwise try args.dataset_dir folder
    pkl_path = args.input_pkl
    if not os.path.exists(pkl_path):
        possible_path = os.path.join(args.dataset_dir, args.input_pkl)
        if os.path.exists(possible_path):
            pkl_path = possible_path
            
    inference_dataset = InferenceDataset(
        config,
        paths=[pkl_path],
        override_num_views=num_images,
    )
    inference_loader = torch.utils.data.DataLoader(
        inference_dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        collate_fn=inference_dataset.custom_collate,
    )
    with torch.no_grad():
        print(f"Starting inference loop for {len(inference_loader)} batches...")
        for batch in tqdm(inference_loader, desc="Inference Batches"):
            print(f"\n" + "="*50)
            print(f"PROCESSING BATCH: {batch['name']}")
            print(f"="*50)
            
            batch = InferenceDataset.move_batch_to_device(
                batch, device, dtype=torch.float16
            )
            print(f"[DEBUG] Batch items moved to {device}")
            for k, v in batch.items():
                if hasattr(v, 'shape'):
                    print(f"  - {k}: shape={v.shape}")
                elif isinstance(v, list):
                    print(f"  - {k}: list, length={len(v)}")
            
            VRAM.print_vram_usage("Start of Batch processing")
            
            # Offloading Logic
            offload_mode = VRAM.should_offload()
            precomputed = None
            
            if offload_mode:
                print(">> Offload Mode: Pre-computing embeddings...")
                # Ensure text features are in batch if needed
                if "text" in model.input_types:
                    batch["t5_text"], batch["clip_text"] = text_feature_extractor(batch["caption"])
                
                precomputed = model.get_condition_embeddings(batch, dtype=torch.float16)
                
                print(">> Offloading Encoders to CPU...")
                if hasattr(model, "dino_ray_extractor"): model.dino_ray_extractor.to("cpu")
                if hasattr(model, "simple_t5_projection"): model.simple_t5_projection.to("cpu")
                if hasattr(model, "simple_clip_projection"): model.simple_clip_projection.to("cpu")
                # text_feature_extractor is separate
                text_feature_extractor.to("cpu") 
                torch.cuda.empty_cache()
                gc.collect()
                VRAM.print_vram_usage("After Offload")

            latents_pred = model.infer_latents(
                batch,
                token_shape=token_shape,
                text_feature_extractor=text_feature_extractor if not offload_mode else None, # Don't use it if offloaded
                num_steps=num_steps,
                use_shifted_sampling=use_shifted_sampling,
                precomputed_embeddings=precomputed
            )
            VRAM.print_vram_usage("After Flow Matching")
            
            # Reload if needed (simple approach: just put them back to device if multiple batches)
            if offload_mode and len(inference_loader) > 1:
                print(">> Reloading modules for next batch...")
                if hasattr(model, "dino_ray_extractor"): model.dino_ray_extractor.to(device)
                if hasattr(model, "simple_t5_projection"): model.simple_t5_projection.to(device)
                if hasattr(model, "simple_clip_projection"): model.simple_clip_projection.to(device)
                text_feature_extractor.to(device)
                
            mesh = vae.infer_mesh_from_latents(latents_pred)[0]
            VRAM.print_vram_usage("After VAE Decoding")
            if args.save_visualization:
                print(f"[DEBUG] Generating visualization reports for {batch['name'][0]}...")
                vis_prd_mesh = mesh.copy()
                if "vertices" in batch:
                    print(f"  - Ground Truth mesh found, adding to visualization.")
                    vis_tgt_mesh = trimesh.Trimesh(
                        vertices=batch["vertices"][0],
                        faces=batch["faces"][0],
                    )
                else:
                    print(f"  - No Ground Truth mesh (real video mode).")
                    vis_tgt_mesh = None
                
                vis_points = batch["semi_dense_points_orig"][0]
                vis_images = batch["images"][0].float().cpu().numpy()
                vis_masks = batch["images"][0].float().cpu().clone().numpy()
                vis_masks[:, 1, :, :] = batch["masks_ingest"][0].float().cpu().numpy()

                save_path = os.path.join(output_dir, f"VIS__{batch['name'][0]}.jpg")
                visualize_prediction_and_groundtruth(
                    vis_prd_mesh,
                    vis_tgt_mesh,
                    vis_points,
                    vis_images,
                    vis_masks,
                    batch["caption"][0],
                    sample_name=batch["name"][0],
                    save_path=save_path,
                )
                print(f"  - Visualization report saved: {save_path}")
            # remove floating geometry, keeping only the largest component
            # sometimes not the best way, but usually works out okay most of the time

            if args.remove_floating_geometry:
                mesh = remove_floating_geometry(mesh)
            # simplify the mesh otherwise it will be too large if you mesh it at 128x128x128 resolution
            if args.simplify_mesh:
                mesh = mesh.simplify_quadric_decimation(face_count=125000)
            # rescale back to the original scale
            print(f"[DEBUG] Rescaling mesh back to original bounds...")
            mesh = inference_dataset.rescale_back(
                batch["index"][0], mesh, args.do_transform_to_world
            )
            
            # Use 'mesh_directory' in project root
            mesh_output_dir = Path("mesh_directory")
            mesh_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use a local temporary path that works on Windows
            tmp_dir = os.path.join(os.getcwd(), args.dataset_dir, "temp")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_output_path_mesh = os.path.join(tmp_dir, f"{batch['name'][0]}_temp.obj")
            
            print(f"[DEBUG] Exporting final .glb to {mesh_output_dir}...")
            mesh.export(tmp_output_path_mesh)
            # convert to glb
            mesh = trimesh.load(tmp_output_path_mesh, force="mesh")
            final_path = mesh_output_dir / (batch["name"][0] + ".glb")
            mesh.export(final_path, include_normals=True)
            print(f"SUCCESS: Result for {batch['name'][0]} saved to {final_path}")
            
            # --- Blender Rendering ---
            mesh_video_dir = Path("mesh_video")
            mesh_video_dir.mkdir(parents=True, exist_ok=True)
            video_output_path = mesh_video_dir / (batch["name"][0] + ".mp4")
            
            print(f"[DEBUG] Rendering video of the mesh using Blender...")
            try:
                import subprocess
                render_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_blender.py")
                cmd = ["blender", "-b", "-P", render_script, "--", str(final_path), str(video_output_path)]
                subprocess.run(cmd, check=True)
                print(f"SUCCESS: Video rendered to {video_output_path}")
            except Exception as e:
                print(f"WARNING: Failed to render video. Is Blender installed and in PATH? Error: {e}")
                
            print(f"="*50 + "\n")


if __name__ == "__main__":
    main()
