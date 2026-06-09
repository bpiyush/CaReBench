#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FEAT_DIR="${FEAT_DIR:-/work/piyush/experiments/MatterOfTime/features}"
VIZ_DIR="$REPO_ROOT/mllm4emb/matteroftime_viz"

python mllm4emb/build_matteroftime_viz_data.py --feat_dir "$FEAT_DIR" --out_dir "$VIZ_DIR" "$@"

echo ""
echo "Open: file://$VIZ_DIR/index.html"
echo "Or:   cd $VIZ_DIR && python -m http.server 8766"
