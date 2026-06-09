#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

python mllm4emb/build_year_embeddings_viz_data.py "$@"

VIZ_DIR="$REPO_ROOT/mllm4emb/year_embeddings_viz"
echo ""
echo "Open in browser:"
echo "  file://$VIZ_DIR/index.html"
echo ""
echo "Or serve locally:"
echo "  cd $VIZ_DIR && python -m http.server 8765"
echo "  → http://localhost:8765"
