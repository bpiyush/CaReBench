#!/bin/bash

# Usage: bash scripts/run_modify_edit_instruction.sh --ngpus 8 --si 0 --ei 20000 --attn sdpa

# Default values
NGPUS=8
SI=0
EI=100000
ATTN="flash_attention_2"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ngpus) NGPUS="$2"; shift 2 ;;
        --si) SI="$2"; shift 2 ;;
        --ei) EI="$2"; shift 2 ;;
        --attn) ATTN="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Calculate chunk size
TOTAL=$((EI - SI))
CHUNK_SIZE=$(( (TOTAL + NGPUS - 1) / NGPUS ))  # Ceiling division

echo "Running with $NGPUS GPUs"
echo "Total range: $SI to $EI ($TOTAL samples)"
echo "Chunk size: $CHUNK_SIZE"
echo "Attention: $ATTN"
echo "----------------------------------------"

# Launch parallel jobs
for ((i=0; i<NGPUS; i++)); do
    START=$((SI + i * CHUNK_SIZE))
    END=$((START + CHUNK_SIZE))
    
    # Ensure END doesn't exceed EI
    if [ $END -gt $EI ]; then
        END=$EI
    fi
    
    # Skip if START >= EI (can happen with small datasets)
    if [ $START -ge $EI ]; then
        continue
    fi
    
    echo "GPU $i: processing indices $START to $END"
    CUDA_VISIBLE_DEVICES=$i python tasks/modify_edit_instruction.py --si $START --ei $END --attn $ATTN &
done

# Wait for all background jobs to complete
wait
echo "----------------------------------------"
echo "All jobs completed!"

