#!/bin/bash
set -e

# Define target directory
TARGET_DIR="/mnt/shared_models/_home/pshrestha"
BLENDER_URL="https://mirrors.dotsrc.org/blender/release/Blender4.2/blender-4.2.0-linux-x86_64.tar.xz"
BLENDER_ARCHIVE="blender_portable.tar.xz"
BLENDER_FOLDER="blender-4.2.0-linux-x86_64"

echo "===================================================="
echo "Installing Portable Blender (No Sudo Required)"
echo "Target: $TARGET_DIR"
echo "===================================================="

# 1. Create directory if it doesn't exist
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR"

# 2. Download Blender
if [ ! -d "$BLENDER_FOLDER" ]; then
    echo "-> Downloading Blender 4.2 LTS..."
    wget "$BLENDER_URL" -O "$BLENDER_ARCHIVE"

    echo "-> Extracting..."
    tar -xvf "$BLENDER_ARCHIVE"
    
    echo "-> Cleanup..."
    rm "$BLENDER_ARCHIVE"
else
    echo "-> Blender folder already exists, skipping download."
fi

# 3. Add to PATH automatically in .bashrc
echo "-> Adding Blender to your PATH in ~/.bashrc..."

# Check if already in PATH to avoid duplicates
if ! grep -q "$BLENDER_FOLDER" ~/.bashrc; then
    echo "export PATH=\"$TARGET_DIR/$BLENDER_FOLDER:\$PATH\"" >> ~/.bashrc
    echo "-> Successfully added to .bashrc"
else
    echo "-> Already present in .bashrc"
fi

echo ""
echo "===================================================="
echo "SUCCESS! Blender installation complete."
echo "===================================================="
echo "To activate it in this session, run:"
echo "  source ~/.bashrc"
echo ""
echo "Then verify by running:"
echo "  blender --version"
echo "===================================================="
