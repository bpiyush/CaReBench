#!/bin/bash
set -euo pipefail

STAGE=${1:-overfit}
BASE_MODEL=${BASE_MODEL:-/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115}
# Wan2.2 default; for LTX2 run: scripts/remap_chiral_pairs_csv_to_ltx2.py then set
# CSV_PATH to data/generated-chiral-pairs-v2-ltx2-complete.csv
CSV_PATH=${CSV_PATH:-/users/piyush/projects/CaReBench/data/generated-chiral-pairs-v2.csv}
OUTPUT_ROOT=${OUTPUT_ROOT:-/work/piyush/experiments/CaRe/Tarsier2-7b-0115-vlemb}
GPUS=${GPUS:-8}
NUM_NODES=${NUM_NODES:-1}

WANDB_PROJECT=${WANDB_PROJECT:-CaReBench-Tarsier2-VLEmb}
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
    LR=1e-6
    MAX_SAMPLES=4
    OVERFIT_NUM_ROWS=4
    OVERFIT_REPEAT=512
    ;;
  smoke)
    MICRO_BATCH_SIZE=1
    BATCH_SIZE=32
    EPOCHS=1
    LR=2e-6
    MAX_SAMPLES=256
    OVERFIT_NUM_ROWS=-1
    OVERFIT_REPEAT=1
    ;;
  full)
    MICRO_BATCH_SIZE=1
    BATCH_SIZE=32
    EPOCHS=2
    # EPOCHS=5
    LR=2e-6
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

# Override stage defaults without editing this file, e.g.:
#   LR_OVERRIDE=5e-6 bash scripts/train_tarsier2_vlemb.sh overfit
#   LR_SCHEDULER=cosine WARMUP_RATIO=0.03 bash scripts/train_tarsier2_vlemb.sh full
# Valid LR_SCHEDULER values: constant, constant_with_warmup, linear, cosine,
# cosine_with_restarts, polynomial, inverse_sqrt, reduce_lr_on_plateau.
if [ -n "${LR_OVERRIDE:-}" ]; then
  LR="$LR_OVERRIDE"
fi
LR_SCHEDULER=${LR_SCHEDULER:-constant}
WARMUP_RATIO=${WARMUP_RATIO:-0.0}

RUN_NAME="tarsier2-vlemb-${STAGE}-$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_NAME}"

echo "Stage: $STAGE"
echo "Model: $BASE_MODEL"
echo "Data: $CSV_PATH"
echo "Output: $OUTPUT_DIR"
echo "WANDB_PROJECT: $WANDB_PROJECT"
echo "Batching: micro=${MICRO_BATCH_SIZE} global=${BATCH_SIZE}"
echo "Learning rate: ${LR} (set LR_OVERRIDE=... to change)"
echo "LR scheduler: ${LR_SCHEDULER} (warmup_ratio=${WARMUP_RATIO})"

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
  --report_to_wandb True

echo "Training finished. Final-only checkpoint is under: ${OUTPUT_DIR}"
