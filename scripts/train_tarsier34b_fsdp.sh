#!/bin/bash

# Fine-tune Tarsier-34B using FSDP (Fully Sharded Data Parallel)
# This script uses PyTorch FSDP to shard the 34B model across multiple GPUs

split=$1
if [ -z "$split" ]; then
    split="nli-27k+ego4d-3k"
fi
echo "Using split: $split"

BASE_MODEL=/work/piyush/pretrained_checkpoints/Tarsier-34b
echo "Using base model: $BASE_MODEL"

# Check if LLM weights exist
LLM_PATH="${BASE_MODEL}-llm"
if [ ! -d "$LLM_PATH" ]; then
    echo "ERROR: LLM weights not found at $LLM_PATH"
    echo "Please run: python tasks/split_weights.py -m $BASE_MODEL"
    exit 1
fi
echo "LLM weights found at: $LLM_PATH"

base_model_name=$(basename $BASE_MODEL)
OUTPUT_DIR="/work/piyush/experiments/CaRe/${base_model_name}-fsdp/${split}"
echo "Using output directory: $OUTPUT_DIR"
mkdir -p $OUTPUT_DIR

RUN_NAME=`basename $OUTPUT_DIR`

# Training hyperparameters for 34B model
BATCH_SIZE=256          # Global batch size
MICRO_BATCH_SIZE=1      # Per-device batch size (very small for 34B to avoid OOM)
EPOCH=1                 # Number of epochs
LR=5e-5                 # Learning rate (lower for full fine-tuning vs LoRA)
WARMUP_RATIO=0.1       # Warmup ratio
CUTOFF_LEN=32          # Maximum sequence length

# Hardware config
GPUS=8                  # Number of GPUs per node
NUM_NODES=1             # Number of nodes

# Data path
CSV_PATH="/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/${split}.csv"

echo "=========================================="
echo "FSDP Training Configuration"
echo "=========================================="
echo "Model: $BASE_MODEL"
echo "Split: $split"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $GPUS"
echo "Batch size: $BATCH_SIZE (micro: $MICRO_BATCH_SIZE)"
echo "Learning rate: $LR"
echo "Epochs: $EPOCH"
echo "=========================================="

# Run training with torchrun (for FSDP)
torchrun \
    --nproc_per_node=$GPUS \
    --nnodes=$NUM_NODES \
    tasks/finetuning_fsdp.py \
    --model_name_or_path $BASE_MODEL \
    --data_path $CSV_PATH \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --num_epochs $EPOCH \
    --learning_rate $LR \
    --warmup_ratio $WARMUP_RATIO \
    --cutoff_len $CUTOFF_LEN \
    --use_neg_sentence \
    --save_steps 100 \
    --logging_steps 10 \
    --grad_checkpoint \
    --seed 42

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Training completed successfully!"
    echo "Model saved to: $OUTPUT_DIR"
    echo "=========================================="
else
    echo "=========================================="
    echo "Training failed!"
    echo "=========================================="
    exit 1
fi

