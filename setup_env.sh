#!/bin/bash
set -e # Exit immediately if a command fails

echo "======================================"
echo "ShapeR Environment Setup Script"
echo "======================================"

# 1. Environment Safety Check
if [[ "$CONDA_DEFAULT_ENV" != "shape" ]]; then
    echo "Warning: Conda environment 'shape' is not currently active!"
    echo "Please run 'conda activate shape' before running this script."
    echo "Press Ctrl+C to abort, or Enter to continue anyway..."
    read -r
fi

# 2. Check CUDA Version and find CUDA_HOME dynamically
echo "-> Checking CUDA installation..."

if ! command -v nvcc &> /dev/null; then
    echo "Error: 'nvcc' could not be found. Please ensure CUDA toolkit is installed and in your PATH."
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo "Found CUDA Version: $CUDA_VERSION"

if [[ "$CUDA_VERSION" != 12.* ]]; then
    echo "Warning: Detected CUDA $CUDA_VERSION, but CUDA 12.8 (or 12.x) was recommended."
    echo "The script will proceed, but you might face issues compiling torchsparse."
    sleep 2
else
    echo "CUDA version $CUDA_VERSION is compatible!"
fi

# Determine CUDA_HOME from nvcc path automatically
NVCC_PATH=$(which nvcc)
export CUDA_HOME=$(dirname $(dirname "$NVCC_PATH"))

# Fallback: if nvcc is just symlinked in /usr/bin but actual cuda is in /usr/local/cuda
if [ "$CUDA_HOME" == "/usr" ] && [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
fi

echo "-> Acquired CUDA_HOME: $CUDA_HOME"

# 3. Export all required CUDA Environment Variables securely
export CUDA_INCLUDE=$CUDA_HOME/include

# Handle systems using lib64 vs lib natively
if [ -d "$CUDA_HOME/lib64" ]; then
    export CUDA_LIB=$CUDA_HOME/lib64
else
    export CUDA_LIB=$CUDA_HOME/lib
fi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CUDA_LIB
export LIBRARY_PATH=$LIBRARY_PATH:$CUDA_LIB
export CFLAGS="-I$CUDA_HOME/include"
export CXXFLAGS="-I$CUDA_HOME/include"
export CPATH="$CUDA_HOME/include:$CPATH"

echo "-> CUDA Environment Variables Exported Successfully."

# 4. Install Conda C++ Libraries
echo "-> Installing system library compilers and sparsehash via Conda..."
conda install -y -c conda-forge gcc_linux-64=11 gxx_linux-64=11 sparsehash

# 5. Install Standard Python Packages
echo "-> Installing standard Python dependencies via pip..."
pip install wheel setuptools ninja
pip install numpy tqdm hydra-core matplotlib opencv-python imageio easydict munch plyfile
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
pip install transformers trimesh scikit-image diffusers gradio peft einops
pip install flash-attn --no-build-isolation --no-cache-dir
pip install "imageio[ffmpeg]" "imageio[pyav]"
pip install pymeshlab sophuspy fast_simplification scikit-learn timm plotly torchdiffeq sentencepiece protobuf pyrender jupyter

# 6. Install Complex Compiling Packages (Torch-Cluster & Torchsparse)
echo "-> Installing Torch-Cluster and Legacy Torchsparse (This may take several minutes)..."
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
pip install --verbose git+https://github.com/nihalsid/torchsparse@legacy --no-build-isolation

# 7. Final Verification
echo "-> Running PyTorch + CUDA SparseTensor Verification Test..."
python -c "
import torch
from torchsparse import SparseTensor
x = SparseTensor(coords=torch.tensor([[1,2,3,0], [4,5,6,1]], dtype=torch.int32), feats=torch.randn(2, 4))
x = x.cuda()
print('\n[SUCCESS] Installation verified! SparseTensor successfully compiled and moved to CUDA.')
"

echo "======================================"
echo "ShapeR Environment Setup Complete!"
echo "======================================"
