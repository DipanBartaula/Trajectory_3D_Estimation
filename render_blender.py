import bpy
import sys
import os
import math
import mathutils

def setup_scene():
    # Clear existing objects
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Add camera
    bpy.ops.object.camera_add(location=(0, -2.5, 0), rotation=(math.radians(90), 0, 0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    # Add lighting (Sun + Area light for soft shadows)
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 5))
    sun = bpy.context.object
    sun.data.energy = 2.0

    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
    area = bpy.context.object
    area.data.energy = 50.0

def load_glb(filepath):
    # Import the GLB file
    bpy.ops.import_scene.gltf(filepath=filepath)
    
    # Get imported objects
    imported_objects = bpy.context.selected_objects
    
    if not imported_objects:
        print("No objects found in GLB.")
        sys.exit(1)
        
    # Calculate bounding box to center and scale the object
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')

    for obj in imported_objects:
        if obj.type == 'MESH':
            for corner in obj.bound_box:
                world_corner = obj.matrix_world @ mathutils.Vector(corner)
                min_x = min(min_x, world_corner.x)
                min_y = min(min_y, world_corner.y)
                min_z = min(min_z, world_corner.z)
                max_x = max(max_x, world_corner.x)
                max_y = max(max_y, world_corner.y)
                max_z = max(max_z, world_corner.z)

    # Find the center of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    center_z = (min_z + max_z) / 2

    # Find the maximum dimension
    size_x = max_x - min_x
    size_y = max_y - min_y
    size_z = max_z - min_z
    max_size = max(size_x, size_y, size_z)

    # Scale the object to fit comfortably in view (e.g. max size = 2.0)
    scale = 2.0 / max_size if max_size > 0 else 1.0

    # Create an empty object to act as the parent for unified rotation/scaling
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    parent_empty = bpy.context.object

    # Parent imported objects to the empty
    for obj in imported_objects:
        obj.parent = parent_empty

    # Apply centering and scaling
    parent_empty.location = (-center_x * scale, -center_y * scale, -center_z * scale)
    parent_empty.scale = (scale, scale, scale)

    return parent_empty

def animate_rotation(obj, frames=120):
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = frames

    # Set initial keyframe (0 degrees)
    obj.rotation_euler = (0, 0, 0)
    obj.keyframe_insert(data_path="rotation_euler", frame=1)

    # Set final keyframe (360 degrees around Z axis, which is UP in Blender)
    # Actually, we want to rotate around the Y axis of the object based on common conventions,
    # but since we adjusted the camera (Y=-2.5 looking at Y=0), rotating Z will rotate the object around its vertical axis.
    obj.rotation_euler = (0, 0, math.radians(360))
    obj.keyframe_insert(data_path="rotation_euler", frame=frames + 1)

    # Set interpolation to linear for smooth looping
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'

def render_video(output_filepath, frames=120):
    scene = bpy.context.scene
    
    # Configure render engine (EEVEE is much faster for this)
    scene.render.engine = 'BLENDER_EEVEE_NEXT' if 'BLENDER_EEVEE_NEXT' in bpy.types.RenderSettings.bl_rna.properties['engine'].enum_items else 'BLENDER_EEVEE'
    
    # Output settings
    scene.render.resolution_x = 512
    scene.render.resolution_y = 512
    scene.render.fps = 30
    
    # Video format settings
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'
    
    # Set output path
    scene.render.filepath = output_filepath

    # Render animation
    bpy.ops.render.render(animation=True)

if __name__ == "__main__":
    # Extract arguments passed after '--'
    if '--' in sys.argv:
        argv = sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = []

    if len(argv) < 2:
        print("Usage: blender -b -P render_blender.py -- <input.glb> <output.mp4>")
        sys.exit(1)

    input_glb = argv[0]
    output_mp4 = argv[1]

    # Run the automated steps
    setup_scene()
    parent_obj = load_glb(input_glb)
    animate_rotation(parent_obj, frames=120)  # 4 seconds at 30 fps
    render_video(output_mp4, frames=120)
