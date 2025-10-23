#!/bin/bash

# MODEL_PATH="/work/piyush/experiments/CaRe/special_milestones/care-stage2-nli-27k-ego4d-3k"
MODEL_PATH="/work/piyush/pretrained_checkpoints/CaRe-7B"
DATA=$1

if [ -z "$DATA" ]; then
    DATA="didemo"
fi

NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Try to raise soft file descriptor limit for this session (no-op if not permitted)
ulimit -n 65535 2>/dev/null || true

accelerate launch \
    --num_machines=1 \
    --num_processes $NUM_GPUS \
    --machine_rank 0 \
    tasks/retrieval.py \
    --model_path $MODEL_PATH \
    --config_path data.json \
    --num_frames 16 \
    --data $DATA