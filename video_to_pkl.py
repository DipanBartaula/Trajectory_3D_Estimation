import argparse
import os
import cv2
import numpy as np
import pickle
import torch
import shutil
from pathlib import Path
from tqdm import tqdm
import io
import pycolmap
from PIL import Image

# VLM and SAM imports
from transformers import BlipProcessor, BlipForConditionalGeneration
from segment_anything import sam_model_registry, SamPredictor

# Set cache directory to checkpoints/huggingface in project root
CACHE_DIR = Path(__file__).parent / "checkpoints" / "huggingface"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def extract_frames(video_path, output_dir, target_fps=2):
    """Extract frames from video to a directory at a specific FPS."""
    print(f"Extracting frames from {video_path} at {target_fps} fps...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []
        
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0 or np.isnan(video_fps):
        video_fps = 30.0 # Fallback
    
    skip_interval = max(1, int(round(video_fps / target_fps)))
    
    frame_count = 0
    saved_count = 0
    frame_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame only if it matches interval
        if frame_count % skip_interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            saved_count += 1
            
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames (from {frame_count} total).")
    if saved_count > 0:
        first_frame = cv2.imread(frame_paths[0])
        print(f"Frame resolution: {first_frame.shape}")
    return frame_paths

def run_sfm(image_dir, output_path):
    """Run Structure from Motion using pycolmap."""
    print("Running SfM with pycolmap...")
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
        
    database_path = output_path / "database.db"
    
    if os.path.exists(database_path):
        os.remove(database_path)
        
    # extract features
    print("Extracting features...")
    pycolmap.extract_features(database_path, image_dir)
    
    # match features
    print("Matching features...")
    pycolmap.match_exhaustive(database_path)
    
    # map
    print("Running incremental mapping...")
    maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)
    
    if not maps:
        raise RuntimeError("SfM failed to reconstruct any model.")
        
    # Return the largest reconstruction
    if maps and 0 in maps:
        print(f"Reconstruction successful. Points: {len(maps[0].points3D)}, Images: {len(maps[0].images)}")
    return maps[0]

def get_caption(image_path, model_id="Salesforce/blip-image-captioning-base"):
    """Generate caption using a small VLM (BLIP)."""
    print("Generating caption...")
    try:
        processor = BlipProcessor.from_pretrained(model_id, cache_dir=str(CACHE_DIR))
        model = BlipForConditionalGeneration.from_pretrained(model_id, cache_dir=str(CACHE_DIR), torch_dtype=torch.float16).to("cuda")
        device = "cuda"
    except torch.cuda.OutOfMemoryError:
        print("Warning: GPU OOM for BLIP. Falling back to CPU for captioning.")
        torch.cuda.empty_cache()
        processor = BlipProcessor.from_pretrained(model_id, cache_dir=str(CACHE_DIR))
        model = BlipForConditionalGeneration.from_pretrained(model_id, cache_dir=str(CACHE_DIR)).to("cpu")
        device = "cpu"
    
    raw_image = Image.open(image_path).convert('RGB')
    dtype = torch.float16 if device == "cuda" else torch.float32
    inputs = processor(raw_image, return_tensors="pt").to(device, dtype)
    
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    print(f"Caption: {caption}")
    
    # Cleanup memory
    del model
    del processor
    torch.cuda.empty_cache()
    
    return caption

def segment_object(image_path, point_prompt=None, device="cuda"):
    """Segment object using SAM."""
    # Use vit_b (base) as it's smaller than hue/large
    sam_checkpoint = "sam_vit_b_01ec64.pth" 
    # Note: User must have the checkpoint. If not present, we might warn or fail.
    # Assuming the user has it or we download it. 
    # For this script, we'll assume the model is loadable via registry if weights are present.
    # If weights are missing, this will fail. We'll try to use a default or assume user provides path.
    
    # For now, let's assume the user has set up SAM or just use the registry logic
    
    # Check if checkpoint exists, if not, maybe download? 
    # To keep it simple, we assume standard usage.
    
    pass # Implemented in the main loop to avoid reloading model


def process_video(video_path, output_pkl, sam_checkpoint=None, device="cuda"):
    """
    Process a video file to create a ShapeR-compatible pickle file.
    """
    # Locate SAM Checkpoint
    if sam_checkpoint is None:
        # Check standard locations
        possible_paths = [
            "sam_vit_b_01ec64.pth",
            "checkpoints/sam_vit_b_01ec64.pth",
            "../sam_vit_b_01ec64.pth"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                sam_checkpoint = p
                break
        if sam_checkpoint is None:
             # Default fallback (might fail if not present)
             sam_checkpoint = "checkpoints/sam_vit_b_01ec64.pth"
             print(f"Warning: SAM checkpoint not explicitly found, trying {sam_checkpoint}")
    # Temp directories
    temp_dir = Path("temp_processing")
    frames_dir = temp_dir / "frames"
    sfm_dir = temp_dir / "sfm"
    
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    # 1. Extract Frames
    frame_paths = extract_frames(video_path, frames_dir)
    if not frame_paths:
        print("No frames extracted.")
        return

    # 2. Run SfM
    try:
        reconstruction = run_sfm(frames_dir, sfm_dir)
    except RuntimeError as e:
        print(f"SfM failed: {e}")
        return
    
    # Extract data from reconstruction
    points3D = []
    for p_id, p in reconstruction.points3D.items():
        points3D.append(p.xyz)
    points3D = np.array(points3D)
    
    if len(points3D) == 0:
        print("No 3D points found.")
        return

    # Center and scale points
    center = np.mean(points3D, axis=0) # Fix: Define center
    points3D_centered = points3D - center
    scale = np.max(np.abs(points3D_centered))
    print(f"3D Model Centered at {center}, Scale factor: {scale}")
    print(f"Total 3D Points: {len(points3D)}")
    
    # 3. Captioning
    # Use the middle frame for captioning
    mid_frame_idx = len(frame_paths) // 2
    caption = get_caption(frame_paths[mid_frame_idx])
    
    # 4. Prepare data needed for Pickle
    rec_images = reconstruction.images
    
    # Lists to store result
    pkl_image_data = []
    pkl_Ts_camera_model = []
    pkl_object_point_projections = []
    pkl_camera_params = []
    pkl_visible_points_model = [] 
    
    point3d_id_to_idx = {p_id: i for i, p_id in enumerate(reconstruction.points3D.keys())}
    points_model_array = np.array([reconstruction.points3D[p_id].xyz for p_id in reconstruction.points3D.keys()])
    
    # SAM Model Load
    print("Loading SAM...")
    sam = sam_model_registry["vit_b"](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    
    sorted_image_ids = sorted(rec_images.keys())
    
    print(f"Processing {len(sorted_image_ids)} frames and segmenting...")
    for img_idx, img_id in enumerate(tqdm(sorted_image_ids)):
        im_obj = rec_images[img_id]
        name = im_obj.name
        frame_path = frames_dir / name
        
        # 1. Read Image Data
        with open(frame_path, "rb") as f:
            img_bytes = f.read()
        pkl_image_data.append(img_bytes)
        
        # 2. Camera Params (Intrinsics)
        cam = reconstruction.cameras[im_obj.camera_id]
        K = np.eye(3)
        if hasattr(cam, "model_name"):
            model_name = cam.model_name
        else:
            # Fallback for newer pycolmap versions where .model is used
            # and .model might be an enum or string
            model_name = cam.model
            if hasattr(model_name, "name"):
                model_name = model_name.name
        
        model_name = str(model_name)
        print(f"DEBUG: Camera model found: {model_name}")

        K = np.eye(3)
        if model_name == "SIMPLE_PINHOLE":
            f_val, cx, cy = cam.params
            K[0,0] = f_val
            K[1,1] = f_val
            K[0,2] = cx
            K[1,2] = cy
        elif model_name == "PINHOLE":
            fx, fy, cx, cy = cam.params
            K[0,0] = fx
            K[1,1] = fy
            K[0,2] = cx
            K[1,2] = cy
        elif model_name == "RADIAL":
             f_val, cx, cy, k1, k2 = cam.params
             K[0,0] = f_val
             K[1,1] = f_val
             K[0,2] = cx
             K[1,2] = cy
        elif model_name == "SIMPLE_RADIAL":
             f_val, cx, cy, k = cam.params
             K[0,0] = f_val
             K[1,1] = f_val
             K[0,2] = cx
             K[1,2] = cy
        
        pkl_camera_params.append(torch.tensor(K, dtype=torch.float32))
        
        # 3. Extrinsics (World to Camera)
        # Handle different pycolmap versions for rotation matrix
        R = np.eye(3)
        if hasattr(im_obj, "rotmat"):
            try:
                R = im_obj.rotmat()
            except TypeError:
                # Some versions might treat rotmat as a property? unlikely if callable failed
                pass
                
        if np.allclose(R, np.eye(3)): # specific check if rotmat failed or wasn't found
             # Check if we can get rotation from cam_from_world
             # Note: in some versions cam_from_world is a property returning Rigid3d
             # in others it might be a method? Or the error implies we accessed something wrong.
             
             rigid3d = None
             if hasattr(im_obj, "cam_from_world"):
                 val = im_obj.cam_from_world
                 # If it's a method, call it
                 if callable(val):
                     try:
                         rigid3d = val()
                     except:
                         pass
                 else:
                     rigid3d = val

             if rigid3d is not None:
                 if hasattr(rigid3d, "rotation"):
                     rot = rigid3d.rotation
                     if hasattr(rot, "matrix"):
                         R = rot.matrix()
                     elif hasattr(rot, "to_matrix"):
                         R = rot.to_matrix()
                     elif callable(rot): # maybe rotation() method
                         try:
                             R = rot().matrix() 
                         except:
                             pass

             if np.allclose(R, np.eye(3)) and hasattr(im_obj, "qvec"):
                  # Fallback: convert qvec to rotmat manually
                  # COLMAP qvec is [w, x, y, z]
                  w, x, y, z = im_obj.qvec
                  R = np.array([
                      [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                      [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
                      [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
                  ])
             elif np.allclose(R, np.eye(3)) and hasattr(im_obj, "rotation_matrix"):
                  R = im_obj.rotation_matrix()

        # Check for translation
        t = np.zeros(3)
        if hasattr(im_obj, "tvec"):
            t = im_obj.tvec
        elif rigid3d is not None and hasattr(rigid3d, "translation"):
             t = rigid3d.translation
        elif hasattr(im_obj, "cam_from_world"):
            val = im_obj.cam_from_world
            # If we didn't get rigid3d above (e.g. earlier if block failed or didn't run)
            # Try to get it again
            r3d = None
            if callable(val):
                try:
                    r3d = val()
                except:
                    pass
            else:
                 r3d = val
            
            if r3d is not None and hasattr(r3d, "translation"):
                 t = r3d.translation

        T_cw = np.eye(4)
        T_cw[:3, :3] = R
        T_cw[:3, 3] = t
        pkl_Ts_camera_model.append(torch.tensor(T_cw, dtype=torch.float32))
        
        # 4. Projections & Visibility
        p2ds = im_obj.points2D 
        
        valid_3d_indices = []
        valid_uvs = []
        
        for p2d in p2ds:
            if p2d.has_point3D():
                pid = p2d.point3D_id
                if pid in point3d_id_to_idx:
                    idx = point3d_id_to_idx[pid]
                    valid_3d_indices.append(idx)
                    valid_uvs.append(p2d.xy)
                    
        pkl_visible_points_model.append(np.array(valid_3d_indices))
        pkl_object_point_projections.append(torch.tensor(np.array(valid_uvs), dtype=torch.float32))
        
        # 5. Segmentation & Filtering
        img_cv2 = cv2.imread(str(frame_path))
        img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)
        
        if len(valid_uvs) > 0:
            uvs = np.array(valid_uvs)
            u_min, v_min = np.min(uvs, axis=0)
            u_max, v_max = np.max(uvs, axis=0)
            box = np.array([u_min, v_min, u_max, v_max])
            
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
            mask = masks[0] 
            
            # Refine visibility
            kept_indices = []
            kept_uvs = []
            
            for i, uv in enumerate(valid_uvs):
                u, v = int(uv[0]), int(uv[1])
                if 0 <= v < mask.shape[0] and 0 <= u < mask.shape[1]:
                    if mask[v, u]:
                        kept_indices.append(valid_3d_indices[i])
                        kept_uvs.append(uv)
            
            pkl_visible_points_model[-1] = np.array(kept_indices)
            pkl_object_point_projections[-1] = torch.tensor(np.array(kept_uvs), dtype=torch.float32)

            if img_idx % 10 == 0:
                print(f"  Frame {img_idx}: valid_uvs={len(valid_uvs)}, kept_after_mask={len(kept_uvs)}")
            
    # Construct final Pickle Dict
    data = {
        "points_model": torch.tensor(points_model_array, dtype=torch.float32),
        "bounds": torch.tensor(scale, dtype=torch.float32),
        "inv_dist_std": torch.zeros(len(points_model_array), dtype=torch.float32),
        "dist_std": torch.zeros(len(points_model_array), dtype=torch.float32),
        "image_data": pkl_image_data,
        "Ts_camera_model": pkl_Ts_camera_model,
        "object_point_projections": pkl_object_point_projections,
        "camera_params": pkl_camera_params,
        "caption": caption,
        "category": "object",
        "T_model_world": torch.eye(4, dtype=torch.float32),
        "T_zup_obj": torch.eye(4, dtype=torch.float32),
        "visible_points_model": pkl_visible_points_model
    }
    
    print(f"Saving to {output_pkl}...")
    with open(output_pkl, "wb") as f:
        pickle.dump(data, f)
    
    print("Video processing done!")

def main():
    parser = argparse.ArgumentParser(description="Convert Video to Input PKL for ShapeR")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input .mp4 video")
    parser.add_argument("--output_pkl", type=str, required=True, help="Path to output .pkl file")
    parser.add_argument("--sam_checkpoint", type=str, default="sam_vit_b_01ec64.pth", help="Path to SAM checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    process_video(args.video_path, args.output_pkl, args.sam_checkpoint, args.device)


if __name__ == "__main__":
    main()
