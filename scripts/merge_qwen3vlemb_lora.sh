#!/bin/bash
# Merge LoRA adapter into the base Qwen3-VL-Embedding checkpoint.
# Writes merged weights to <LORA_DIR>/merged_checkpoint/ (no --save_dir needed).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BASE_MODEL=/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B
# Optional: pass a different LoRA run dir as first argument
LORA_DIR="${1:-/work/piyush/experiments/CaRe/Qwen3-VL-Embedding-8B-lora/covr/chiral10k-covr10k}"

echo "Base model:     $BASE_MODEL"
echo "LoRA run dir:   $LORA_DIR"
echo "Merged output:  $LORA_DIR/merged_checkpoint/"
echo ""

python tasks/merge_weights_qwen3vlemb_lora.py \
  -b "$BASE_MODEL" \
  -f "$LORA_DIR"

echo "Done."
