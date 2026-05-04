#!/usr/bin/env bash
# Merge every Hugging Face-style checkpoint-* (LLM-only) under a run directory into
# full Tarsier2 MLLM checkpoints via tasks/merge_weights_tarsier2.py.
#
# Example:
#   ./scripts/merge_tarsier2_stepwise_checkpoints.sh \
#     -b /work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/ \
#     -m /work/piyush/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k-stepwise/
#
# Each checkpoint-N is written to checkpoint-N/merged_checkpoint (same as running the
# Python script with -f pointing at that directory).
# Steps whose merged_checkpoint already looks complete (config + weights) are skipped.

set -euo pipefail

# True if merged MLLM output dir appears finished (not empty / partial).
merged_mllm_complete() {
  local dir="$1"
  [[ -d "$dir" && -f "$dir/config.json" ]] || return 1
  [[ -f "$dir/model.safetensors" || -f "$dir/pytorch_model.bin" ]] && return 0
  compgen -G "$dir/model-*-of-*.safetensors" >/dev/null && return 0
  return 1
}

usage() {
  cat <<'EOF'
Merge every checkpoint-* under a run dir into full Tarsier2 MLLM checkpoints.

Usage:
  merge_tarsier2_stepwise_checkpoints.sh -b BASE_MODEL -m MODEL_DIR

Options:
  -b, --base-model PATH     Base Tarsier2 MLLM directory (required).
  -m, --model-dir PATH      Run folder containing checkpoint-* subdirs (required).
  -p, --python CMD          Python to use (default: python).
  --merge-script PATH       Path to merge_weights_tarsier2.py (default: <repo>/tasks/merge_weights_tarsier2.py).
  -h, --help                Show this help.

Requires conda env carebench (same as merge_weights_tarsier2.py).
Each checkpoint-N produces checkpoint-N/merged_checkpoint.
If that directory already has config.json and model weight files, the step is skipped.
EOF
  exit "${1:-0}"
}

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$THIS_DIR/.." && pwd)"
BASE_MODEL=""
MODEL_DIR=""
PYTHON_BIN="${PYTHON_BIN:-python}"
MERGE_SCRIPT="${MERGE_SCRIPT:-$REPO_ROOT/tasks/merge_weights_tarsier2.py}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--base-model)
      BASE_MODEL="${2:?}"
      shift 2
      ;;
    -m|--model-dir)
      MODEL_DIR="${2:?}"
      shift 2
      ;;
    -p|--python)
      PYTHON_BIN="${2:?}"
      shift 2
      ;;
    --merge-script)
      MERGE_SCRIPT="${2:?}"
      shift 2
      ;;
    -h|--help)
      usage 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage 1
      ;;
  esac
done

if [[ -z "$BASE_MODEL" || -z "$MODEL_DIR" ]]; then
  echo "Error: -b/--base-model and -m/--model-dir are required." >&2
  usage 1
fi

if [[ ! -d "$BASE_MODEL" ]]; then
  echo "Error: base model path is not a directory: $BASE_MODEL" >&2
  exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Error: model dir is not a directory: $MODEL_DIR" >&2
  exit 1
fi

if [[ ! -f "$MERGE_SCRIPT" ]]; then
  echo "Error: merge script not found: $MERGE_SCRIPT" >&2
  exit 1
fi

if [[ "${CONDA_DEFAULT_ENV:-}" != "carebench" ]]; then
  echo "Warning: CONDA_DEFAULT_ENV is '${CONDA_DEFAULT_ENV:-}' (expected carebench). merge_weights_tarsier2.py will fail if wrong env." >&2
fi

# Collect checkpoint-* dirs, sort by numeric step (not lexicographic).
mapfile -t CHECKPOINTS < <(
  shopt -s nullglob
  for d in "$MODEL_DIR"/checkpoint-*; do
    [[ -d "$d" ]] || continue
    step="${d##*checkpoint-}"
    [[ "$step" =~ ^[0-9]+$ ]] || continue
    printf '%s\t%s\n' "$step" "$d"
  done | sort -n | cut -f2-
)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "Error: no checkpoint-* directories under: $MODEL_DIR" >&2
  exit 1
fi

TOTAL="${#CHECKPOINTS[@]}"
START_TS=$(date +%s)
RAN=0
SKIPPED=0

divider() { printf '%s\n' "--------------------------------------------------------------------------------"; }

echo ""
divider
echo "  Tarsier2 stepwise merge"
echo "  Base model:     $BASE_MODEL"
echo "  Run directory:  $MODEL_DIR"
echo "  Checkpoints:    $TOTAL"
echo "  Merge script:   $MERGE_SCRIPT"
echo "  Python:         $PYTHON_BIN"
divider
echo ""

idx=0
for ckpt in "${CHECKPOINTS[@]}"; do
  ((idx++)) || true
  name="$(basename "$ckpt")"
  out="${ckpt}/merged_checkpoint"
  echo ""
  divider
  echo "  [$idx / $TOTAL]  $name"
  echo "  LLM weights:    $ckpt"
  echo "  Merged output:  $out"
  divider
  if merged_mllm_complete "$out"; then
    ((SKIPPED++)) || true
    echo ""
    echo "  [skip] Merged checkpoint already present (config + weights)."
    continue
  fi
  SECONDS=0
  if "$PYTHON_BIN" "$MERGE_SCRIPT" -b "$BASE_MODEL" -f "$ckpt"; then
    ((RAN++)) || true
    echo ""
    echo "  ✓ Done in ${SECONDS}s → $out"
  else
    echo ""
    echo "  ✗ Failed on $name (exit $?)" >&2
    exit 1
  fi
done

END_TS=$(date +%s)
ELAPSED=$((END_TS - START_TS))

echo ""
divider
echo "  Finished in ${ELAPSED}s:  merged $RAN  |  skipped $SKIPPED  |  total steps $TOTAL"
divider
echo ""
