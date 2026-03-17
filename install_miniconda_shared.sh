#!/bin/bash
set -e

# 1. Define Paths
# We will install Miniconda itself into this shared folder
TARGET_DIR="/mnt/shared_models/_home/pshrestha"
MINICONDA_ROOT="$TARGET_DIR/miniconda3"
INSTALLER_SCRIPT="miniconda_installer.sh"

echo "===================================================="
echo "Installing Miniconda to Shared Directory"
echo "Target: $MINICONDA_ROOT"
echo "===================================================="

# 2. Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# 3. Download Miniconda for Linux
echo "-> Downloading latest Miniconda installer..."
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$INSTALLER_SCRIPT"

# 4. Install in Batch Mode (-b) to the custom path (-p)
echo "-> Running installation (this may take a moment)..."
bash "$INSTALLER_SCRIPT" -b -p "$MINICONDA_ROOT"

# 5. Cleanup Installer
rm "$INSTALLER_SCRIPT"

# 6. Initialize for the current shell
echo "-> Initializing shell for the new installation..."
source "$MINICONDA_ROOT/bin/activate"
conda init bash

# 7. Create the 'shape' environment inside the shared installation
echo "-> Creating 'shape' environment with Python 3.10..."
"$MINICONDA_ROOT/bin/conda" create -n shape python=3.10 -y

echo ""
echo "===================================================="
echo "SUCCESS! Miniconda and 'shape' env are ready."
echo "Path: $MINICONDA_ROOT"
echo "===================================================="
echo "Instructions:"
echo "1. Run: source ~/.bashrc"
echo "2. Run: conda activate shape"
echo "3. You can then run your setup_env.sh script."
echo "===================================================="
