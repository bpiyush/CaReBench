#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

eval "$(conda shell.bash hook)"
conda activate qwen

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

DATA_ROOT="${DATA_ROOT:-/scratch/shared/beegfs/piyush/datasets/Time10K}"
OUT_DIR="${OUT_DIR:-/work/piyush/experiments/MatterOfTime/features}"

exec python mllm4emb/compute_matteroftime_features.py \
  --data_root "$DATA_ROOT" \
  --out_dir "$OUT_DIR" \
  "$@"
