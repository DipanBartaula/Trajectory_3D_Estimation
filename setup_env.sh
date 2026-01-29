#!/bin/bash

# Configuration
ENV_NAME="shaper"
PYTHON_VERSION="3.10"

echo "Using system CUDA configuration (skipping explicit CUDA exports)..."

# Step 1: Create Conda Environment
echo "Creating Conda environment '$ENV_NAME'..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# initialize conda in bash script
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment. Please check your conda installation."
    exit 1
fi

# Step 3: Install Python Packages
echo "Installing compilers and dependencies..."
conda install -c conda-forge gcc_linux-64=11 gxx_linux-64=11 -y
conda install -c conda-forge sparsehash -y

echo "Installing pip packages..."
pip install wheel setuptools ninja
pip install numpy tqdm hydra-core matplotlib opencv-python imageio easydict munch plyfile
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers trimesh scikit-image diffusers gradio peft einops
pip install flash-attn --no-build-isolation --no-cache-dir
pip install "imageio[ffmpeg]" "imageio[pyav]"
pip install pymeshlab sophuspy fast_simplification scikit-learn timm plotly torchdiffeq sentencepiece protobuf pyrender jupyter

# Step 4: Install Torch-Cluster and Torchsparse
echo "Installing torch-cluster..."
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.1+cu128.html

echo "Installing torchsparse (this may take a while)..."
pip install --verbose git+https://github.com/nihalsid/torchsparse@legacy --no-build-isolation

# Step 5: Install Additional Tools for Video Processing (SfM & VLM/SAM)
echo "Installing pycolmap and segment-anything..."
pip install pycolmap segment-anything

echo "Downloading SAM checkpoint (vit_b)..."
if command -v wget &> /dev/null; then
    wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
else
    echo "wget not found. Please manually download sam_vit_b_01ec64.pth from https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
fi

echo "Setup complete! Activate the environment with: conda activate $ENV_NAME"
