#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate carebench
export PYTHONPATH=$PWD

BASE="/work/piyush/experiments/CaRe/Tarsier2-7b-0115/ablations"
PATTERN="_nuanced_retrieval_data-validation-v1_embeddings.pt"

for sub in "$BASE"/*/; do
    sub_name=$(basename "$sub")
    match=$(ls "$sub/merged_checkpoint/embs/"*"$PATTERN" 2>/dev/null)
    if [ -z "$match" ]; then
        echo "SKIP (no embeddings): $sub_name"
        continue
    fi

    fname=$(basename "$match")
    model_name="${fname%$PATTERN}"
    csv_name="nuanced_retrieval_data-validation-v1"
    result_file="$BASE/$sub_name/merged_checkpoint/metrics/metrics_${model_name}_${csv_name}.json"

    if [ -f "$result_file" ]; then
        echo "SKIP (results exist): $sub_name"
        continue
    fi

    echo "========================================"
    echo "RUN: $sub_name (model_name=$model_name)"
    echo "========================================"
    python evals_tarsier2/compute_metrics_validation.py \
        --model_path "$BASE/$sub_name/merged_checkpoint/" \
        --model_name "$model_name"
    echo ""
done
