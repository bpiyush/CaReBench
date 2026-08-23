#!/usr/bin/env bash
# Interactive (foreground) launch of the FastAPI demo.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-/users/piyush/miniconda3/envs/carebench/bin/python}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
cd "$ROOT"
exec "$PYTHON" -u server.py "$@"
