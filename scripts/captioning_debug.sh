#!/bin/bash

# MODEL_PATH="/work/piyush/experiments/CaRe/special_milestones/care-stage2-nli-27k-ego4d-3k"
# SAVE_DIR="/work/piyush/experiments/CaRe/results/captioning_carebench/care-stage2-nli-27k-ego4d-3k"

MODEL_PATH=/work/piyush/experiments/CaRe/qwen2vl/debug_run_nli-275k/merged_checkpoint
SAVE_DIR="/work/piyush/experiments/CaRe/results/captioning_carebench/qwen2vl/debug_run_nli-275k"

mkdir -p $SAVE_DIR
DATA=carebench

accelerate launch \
    --num_machines=1 \
    --num_processes 2 \
    --machine_rank 0 \
    tasks/captioning.py \
    --config_path data.json \
    --dataset_name $DATA \
    --model_path $MODEL_PATH \
    --save_dir $SAVE_DIR \
    --num_frames 32 \
    --api_endpoint "https://api.deepseek.com/v1" \
    --api_key "$DEEPSEEK_API_KEY" \
    --api_model "deepseek-chat" \
    --api_num_worker 64 \
    --evaluate
