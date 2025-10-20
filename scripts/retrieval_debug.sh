#!/bin/bash

MODEL_PATH="/work/piyush/experiments/CaRe/special_milestones/care-stage2-nli-27k-ego4d-3k"
DATA=didemo

# Try to raise soft file descriptor limit for this session (no-op if not permitted)
ulimit -n 65535 2>/dev/null || true

accelerate launch \
    --num_machines=1 \
    --num_processes 8 \
    --machine_rank 0 \
    tasks/retrieval.py \
    --model_path $MODEL_PATH \
    --config_path data.json \
    --num_frames 16 \
    --data $DATA