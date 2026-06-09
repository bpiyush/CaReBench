#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

eval "$(conda shell.bash hook)"
conda activate qwen

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python -c "import transformers; from packaging.version import Version; assert Version(transformers.__version__) >= Version('4.57.0'), f'Need transformers>=4.57.0, got {transformers.__version__}'"
python -c "from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel"

exec python mllm4emb/embed_year_qwen3vl.py "$@"
