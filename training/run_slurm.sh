#!/bin/bash
#SBATCH --job-name=shaper_train
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # One task launching the script
#SBATCH --gres=gpu:4            # Request 4 GPUs
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# Create logs dir if not exists
mkdir -p logs

# Export Network/Env variables if needed
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=1 # Sometimes needed on some clusters
export WANDB_PROJECT="shaper-lora-training"

# Activate your environment here
# source activate shaper

echo "Starting training on $SLURM_JOB_NODELIST"
echo "GPUs available: $CUDA_VISIBLE_DEVICES"

# Launch training with Accelerate
# --multi_gpu enables DDP
# --num_processes should match GPU count
accelerate launch --multi_gpu --num_processes 4 training/train.py
