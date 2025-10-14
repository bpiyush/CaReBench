#!/bin/bash

OUTPUT_DIR="/work/piyush/experiments/CaRe/debug_run"
RUN_NAME=`basename $OUTPUT_DIR`

args=()

BASE_MODEL="/work/piyush/pretrained_checkpoints/CaRe-7B-Stage-1"
# BATCH_SIZE=768
# MICRO_BATCH_SIZE=32
BATCH_SIZE=16
MICRO_BATCH_SIZE=2
EPOCH=2
LR=2e-5
WARMUP_RATIO=0.1
CUTOFF_LEN=32
GPUS=8
NUM_NODES=1

# CSV_PATH='/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/nli_for_simcse.csv'
CSV_PATH='/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/nli_for_simcse-10k.csv'

echo $BASE_MODEL
echo $MICRO_BATCH_SIZE $BATCH_SIZE
wandb online

deepspeed --num_gpus=$GPUS --num_nodes=$NUM_NODES tasks/finetuning.py \
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
        --use_neg_sentence --save_steps 1000 \
        --deepspeed ds.config \
        --bf16 \
        --logging_steps 1 --grad_checkpoint