#!/bin/bash
# Script to set Hugging Face Token environment variable
# Usage: source bash/set_hf_token.sh <YOUR_TOKEN>

if [ -z "$1" ]; then
    echo "Usage: source bash/set_hf_token.sh <YOUR_HF_TOKEN>"
else
    export HF_TOKEN="$1"
    echo "HF_TOKEN has been set."
fi
