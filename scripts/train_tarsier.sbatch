#!/bin/bash
#SBATCH --job-name=train_tarsier
#SBATCH --partition=low-prio-gpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=300000
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --constraint=a40
#SBATCH --output=logs/train_tarsier_%j.out
#SBATCH --error=logs/train_tarsier_%j.err

# Change to project directory
cd /users/piyush/projects/CaReBench

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate carebench

# Set PYTHONPATH
export PYTHONPATH=$PWD

# Create logs directory if it doesn't exist
mkdir -p logs

# Run the training script with the specified split
bash scripts/train_tarsier.sh final-10112025/before-after-random_pairs-4000
