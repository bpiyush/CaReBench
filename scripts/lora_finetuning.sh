#!/bin/bash

split=$1
if [ -z "$split" ]; then
    split="nli-27k+ego4d-3k"
fi

OUTPUT_DIR="/work/piyush/experiments/CaRe/lora/debug_run_${split}"
# OUTPUT_DIR="/work/piyush/experiments/CaRe/qwen2vl/lora_run_${split}"
RUN_NAME=`basename $OUTPUT_DIR`

args=()

BASE_MODEL="/work/piyush/pretrained_checkpoints/CaRe-7B-Stage-1"
# BASE_MODEL="/work/piyush/pretrained_checkpoints/Qwen2-VL-7B-Instruct"

# BATCH_SIZE=768
# MICRO_BATCH_SIZE=32
BATCH_SIZE=32
MICRO_BATCH_SIZE=4
# EPOCH=2
EPOCH=1
LR=2e-4
WARMUP_RATIO=0.1
CUTOFF_LEN=32
GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_NODES=1

# LoRA hyperparameters
LORA_RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,v_proj"  # Common projection layers to apply LoRA

# CSV_PATH='/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/nli-275k.csv'
# CSV_PATH='/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/nli_for_simcse-10k.csv'
# CSV_PATH='/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/nli+ego4d-20k.csv'
# CSV_PATH='/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/nli45k+ego4d-5k.csv'
# CSV_PATH='/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/nli-9k+ego4d-1k.csv'
# CSV_PATH='/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/nli-90k+ego4d-10k.csv'
CSV_PATH="/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/${split}.csv"

echo $BASE_MODEL
echo $MICRO_BATCH_SIZE $BATCH_SIZE
wandb online

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES tasks/lora_finetuning.py \
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
        --use_neg_sentence --save_steps 100000 \
        --deepspeed ds.config \
        --bf16 \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_target_modules "$LORA_TARGET_MODULES" \
        --logging_steps 1 --grad_checkpoint

