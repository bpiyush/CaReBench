#!/bin/bash

split=$1
if [ -z "$split" ]; then
    split="nli-27k+ego4d-3k"
fi
echo "Using split: $split"

BASE_MODEL=/work/piyush/pretrained_checkpoints/Tarsier-7b
echo "Using base model: $BASE_MODEL"

base_model_name=$(basename $BASE_MODEL)
OUTPUT_DIR="/work/piyush/experiments/CaRe/${base_model_name}/${split}"
echo "Using output directory: $OUTPUT_DIR"
RUN_NAME=`basename $OUTPUT_DIR`

args=()

BATCH_SIZE=768
MICRO_BATCH_SIZE=32
# BATCH_SIZE=32
# MICRO_BATCH_SIZE=4
# EPOCH=2
EPOCH=10
# EPOCH=10
# EPOCH=1
LR=2e-5
# LR=2e-4 # paper says 2e-4 but the config had 2e-5
WARMUP_RATIO=0.1
CUTOFF_LEN=32
GPUS=8
NUM_NODES=1
CSV_PATH="/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/${split}.csv"

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
        --use_neg_sentence \
        --save_steps 100000 \
        --deepspeed ds.config \
        --bf16 \
        --logging_steps 1 \
        --grad_checkpoint


# Delete the heavy checkpoint (I anyways save model after training)
echo "Deleting the heavy checkpoint..."
rm -rf $OUTPUT_DIR/checkpoint-*

echo "Done!"