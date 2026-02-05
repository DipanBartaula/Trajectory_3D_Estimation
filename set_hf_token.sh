#!/bin/bash
# Script to set Hugging Face Token environment variable
# Usage: source set_hf_token.sh <YOUR_TOKEN>
# Or edit the file to hardcode the token and run: source set_hf_token.sh

if [ -z "$1" ]; then
    echo "Usage: source set_hf_token.sh <YOUR_HF_TOKEN>"
    echo "Or set the token inside the script."
    # You can also uncomment the line below and paste your token
    # export HF_TOKEN="hf_..."
else
    export HF_TOKEN="$1"
    echo "HF_TOKEN has been set."
fi
