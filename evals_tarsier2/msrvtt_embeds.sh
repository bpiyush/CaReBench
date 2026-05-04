#!/usr/bin/env bash
# Run each checkpoint's merge + MSRVTT embeddings in parallel, up to NUM_GPUS
# jobs at a time; round-robin CUDA_VISIBLE_DEVICES over GPUs.

set -euo pipefail

BASE_CKPT="${BASE_CKPT:-/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/}"
EXP_ROOT="${EXP_ROOT:-/work/piyush/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k-stepwise}"
CSV_PATH="${CSV_PATH:-./data/nuanced_retrieval_data-v1.csv}"
MODEL_NAME="${MODEL_NAME:-tarsier2+tara}"

NUM_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
if [[ -z "$NUM_GPUS" || "$NUM_GPUS" -lt 1 ]]; then
  NUM_GPUS=1
fi
echo "NUM_GPUS=$NUM_GPUS (merge + embed per checkpoint; max ${NUM_GPUS} concurrent)"

n=0
for i in $(seq 40 10 130); do
  g=$((n % NUM_GPUS))
  (
    set -euo pipefail
    echo "=== checkpoint-$i on CUDA_VISIBLE_DEVICES=$g ==="
    # echo python tasks/merge_weights_tarsier2.py -b "$BASE_CKPT" \
    #   -f "${EXP_ROOT}/checkpoint-$i/"
    echo python evals_tarsier2/compute_embeddings.py --model_name "$MODEL_NAME" \
      --model_path "${EXP_ROOT}/checkpoint-$i/merged_checkpoint/" \
      --only_msrvtt --csv_path "$CSV_PATH"
    echo "=== finished checkpoint-$i ==="
  ) &
  ((++n))
  if (( n % NUM_GPUS == 0 )); then
    wait
  fi
done
wait
echo "All checkpoint jobs completed."
