#!/bin/bash

split=$1
if [ -z "$split" ]; then
    split="nli-27k+ego4d-3k"
fi
echo "Using split: $split"

BASE_MODEL=/work/piyush/pretrained_checkpoints/Tarsier-7b
echo "Using base model: $BASE_MODEL"

base_model_name=$(basename $BASE_MODEL)
OUTPUT_DIR="/work/piyush/experiments/CaRe/${base_model_name}-lora/${split}"
echo "Using output directory: $OUTPUT_DIR"
RUN_NAME=`basename $OUTPUT_DIR`

args=()

BATCH_SIZE=768
MICRO_BATCH_SIZE=32
EPOCH=2
LR=2e-4  # Higher LR for LoRA (typical range: 1e-4 to 3e-4)
WARMUP_RATIO=0.1
CUTOFF_LEN=32
GPUS=8
NUM_NODES=1
CSV_PATH="/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/${split}.csv"

# LoRA hyperparameters
LORA_RANK=16           # Rank of LoRA matrices (typical: 8-64)
LORA_ALPHA=32          # Scaling factor (typical: 2x rank)
LORA_DROPOUT=0.05      # Dropout for LoRA layers
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"  # Target attention projections in LLaMA

echo $BASE_MODEL
echo $MICRO_BATCH_SIZE $BATCH_SIZE
echo "LoRA config: rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "Target modules: $LORA_TARGET_MODULES"
wandb online

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES tasks/finetuning_lora.py \
        --model_name_or_path $BASE_MODEL \
        --data_path $CSV_PATH \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE  \
        --num_epochs $EPOCH \
        --warmup_ratio $WARMUP_RATIO \
        --learning_rate $LR \
        --cutoff_len $CUTOFF_LEN \
        --output_dir $OUTPUT_DIR  \
        --run_name $RUN_NAME \
        --use_neg_sentence \
        --save_steps 100000 \
        --deepspeed ds.config \
        --bf16 \
        --logging_steps 1 \
        --grad_checkpoint \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_target_modules "$LORA_TARGET_MODULES"


# Delete the heavy checkpoint (LoRA checkpoints are small, but intermediate ones can be removed)
echo "Deleting intermediate checkpoints..."
rm -rf $OUTPUT_DIR/checkpoint-*

echo "Done! LoRA adapter saved to $OUTPUT_DIR"

