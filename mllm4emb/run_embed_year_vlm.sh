#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

eval "$(conda shell.bash hook)"
conda activate qwen

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -c "import clip"
python -c "from transformers import Siglip2Model"

exec python mllm4emb/embed_year_vlm.py "$@"
