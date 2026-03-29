#!/bin/bash

split=$1
if [ -z "$split" ]; then
    split="covr/chiral10k-covr10k"
fi
echo "Using split: $split"

BASE_MODEL=/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B
echo "Using base model: $BASE_MODEL"

base_model_name=$(basename $BASE_MODEL)
OUTPUT_DIR="/work/piyush/experiments/CaRe/${base_model_name}-lora/${split}"
echo "Using output directory: $OUTPUT_DIR"
RUN_NAME=`basename $OUTPUT_DIR`

BATCH_SIZE=384
MICRO_BATCH_SIZE=4
EPOCH=2
LR=2e-4
WARMUP_RATIO=0.1
CUTOFF_LEN=512
GPUS=8
NUM_NODES=1
CSV_PATH="/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/${split}.csv"

# LoRA hyperparameters
LORA_RANK=16
LORA_ALPHA=32
LORA_DROPOUT=0.05
LORA_TARGET_MODULES="q_proj,k_proj,v_proj,o_proj"

echo $BASE_MODEL
echo $MICRO_BATCH_SIZE $BATCH_SIZE
echo "LoRA config: rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "Target modules: $LORA_TARGET_MODULES"
wandb online

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES tasks/finetuning_qwen3vlemb_lora.py \
        --model_name_or_path $BASE_MODEL \
        --data_path $CSV_PATH \
        --batch_size $BATCH_SIZE \
        --micro_batch_size $MICRO_BATCH_SIZE \
        --num_epochs $EPOCH \
        --warmup_ratio $WARMUP_RATIO \
        --learning_rate $LR \
        --cutoff_len $CUTOFF_LEN \
        --output_dir $OUTPUT_DIR \
        --run_name $RUN_NAME \
        --save_steps 100000 \
        --deepspeed ds.config \
        --bf16 \
        --logging_steps 1 \
        --grad_checkpoint \
        --lora_rank $LORA_RANK \
        --lora_alpha $LORA_ALPHA \
        --lora_dropout $LORA_DROPOUT \
        --lora_target_modules "$LORA_TARGET_MODULES"

# Delete intermediate checkpoints (LoRA adapters are small, but intermediates can be removed)
echo "Deleting intermediate checkpoints..."
rm -rf $OUTPUT_DIR/checkpoint-*

echo "Done! LoRA adapter saved to $OUTPUT_DIR"
