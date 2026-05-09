#!/bin/bash
set -euo pipefail

STAGE=${1:-smoke}
BASE_MODEL=${BASE_MODEL:-/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115}
# Wan2.2 default; for LTX2: remap (scripts/remap_chiral_pairs_csv_to_ltx2.py) then CSV_PATH=.../generated-chiral-pairs-v2-ltx2-complete.csv
CSV_PATH=${CSV_PATH:-/users/piyush/projects/CaReBench/data/generated-chiral-pairs-v2.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/piyush/experiments/CaRe/Tarsier2-7b-0115-vlemb-lora}
GPUS=${GPUS:-8}
NUM_NODES=${NUM_NODES:-1}

LORA_RANK=${LORA_RANK:-16}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj}
# When true (default), LoRA weights are merged into the base MLLM at save time
# so the output directory is a self-contained Tarsier2 checkpoint.
# Set LORA_MERGE=false to save only the adapter.
LORA_MERGE=${LORA_MERGE:-true}

WANDB_PROJECT=${WANDB_PROJECT:-CaReBench-Tarsier2-VLEmb-LoRA}
WANDB_ENTITY=${WANDB_ENTITY:-}
export WANDB_PROJECT
if [ -n "${WANDB_ENTITY}" ]; then
  export WANDB_ENTITY
fi

case "$STAGE" in
  overfit)
    MICRO_BATCH_SIZE=1
    BATCH_SIZE=8
    EPOCHS=20
    LR=2e-4
    MAX_SAMPLES=4
    OVERFIT_NUM_ROWS=4
    OVERFIT_REPEAT=512
    ;;
  smoke)
    MICRO_BATCH_SIZE=1
    BATCH_SIZE=32
    EPOCHS=1
    LR=1e-4
    MAX_SAMPLES=256
    OVERFIT_NUM_ROWS=-1
    OVERFIT_REPEAT=1
    ;;
  full)
    MICRO_BATCH_SIZE=1
    BATCH_SIZE=32
    EPOCHS=2
    LR=1e-4
    MAX_SAMPLES=-1
    OVERFIT_NUM_ROWS=-1
    OVERFIT_REPEAT=1
    ;;
  *)
    echo "Unknown stage: $STAGE"
    echo "Usage: $0 [overfit|smoke|full]"
    exit 1
    ;;
esac

if [ -n "${LR_OVERRIDE:-}" ]; then
  LR="$LR_OVERRIDE"
fi
LR_SCHEDULER=${LR_SCHEDULER:-constant}
WARMUP_RATIO=${WARMUP_RATIO:-0.0}

RUN_NAME="tarsier2-vlemb-lora-${STAGE}-$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

echo "Stage: $STAGE"
echo "Model: $BASE_MODEL"
echo "Data: $CSV_PATH"
echo "Output: $OUTPUT_DIR"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "Batching: micro=${MICRO_BATCH_SIZE} global=${BATCH_SIZE}"
echo "Learning rate: ${LR} (set LR_OVERRIDE=... to change)"
echo "LR scheduler: ${LR_SCHEDULER} (warmup_ratio=${WARMUP_RATIO})"
echo "LoRA: rank=${LORA_RANK} alpha=${LORA_ALPHA} dropout=${LORA_DROPOUT} targets=${LORA_TARGET_MODULES} merge=${LORA_MERGE}"

wandb online

deepspeed --num_gpus="${GPUS}" --num_nodes="${NUM_NODES}" tasks/finetuning_tarsier2_vlemb.py \
  --model_name_or_path "${BASE_MODEL}" \
  --data_path "${CSV_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --batch_size "${BATCH_SIZE}" \
  --micro_batch_size "${MICRO_BATCH_SIZE}" \
  --num_epochs "${EPOCHS}" \
  --learning_rate "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --lr_scheduler_type "${LR_SCHEDULER}" \
  --run_name "${RUN_NAME}" \
  --deepspeed ds.config.tarsier2_vlemb.json \
  --bf16 \
  --grad_checkpoint \
  --logging_steps 1 \
  --max_samples "${MAX_SAMPLES}" \
  --overfit_num_rows "${OVERFIT_NUM_ROWS}" \
  --overfit_repeat "${OVERFIT_REPEAT}" \
  --report_to_wandb True \
  --lora True \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_target_modules "${LORA_TARGET_MODULES}" \
  --lora_merge "${LORA_MERGE}"

if [ "${LORA_MERGE}" = "true" ] || [ "${LORA_MERGE}" = "True" ]; then
  echo "Training finished. Merged MLLM checkpoint is under: ${OUTPUT_DIR}"
else
  echo "Training finished. LoRA adapter checkpoint is under: ${OUTPUT_DIR}"
fi
