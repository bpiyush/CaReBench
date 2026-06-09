#!/usr/bin/env python3
"""Export year-embedding projections for the interactive HTML viewer."""
import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


MODEL_REGISTRY = {
    "year_embeddings.pt": {"id": "tara", "name": "TARA"},
    "year_embeddings_qwen3vl.pt": {"id": "qwen3vl", "name": "Qwen3-VL"},
    "year_embeddings_clip.pt": {"id": "clip", "name": "CLIP"},
    "year_embeddings_dinotxt.pt": {"id": "dinotxt", "name": "DINO.txt"},
    "year_embeddings_siglip2.pt": {"id": "siglip2", "name": "SigLIP2"},
}


def load_embeddings(path: str) -> tuple[np.ndarray, List[int], str, str]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    years = [int(y) for y in data["years"]]
    Z = data["embeddings"]
    if torch.is_tensor(Z):
        Z = Z.numpy()
    meta = MODEL_REGISTRY.get(os.path.basename(path), {})
    model_id = meta.get("id") or os.path.splitext(os.path.basename(path))[0]
    model_name = meta.get("name") or data.get("model") or model_id
    return np.asarray(Z, dtype=np.float32), years, model_id, str(model_name)


def project_pca(Z: np.ndarray, n_components: int = 3) -> Dict:
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(Z)
    return {
        "points": coords.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }


def project_tsne(Z: np.ndarray, n_components: int = 3, perplexity: float = 30.0) -> Dict:
    n = len(Z)
    perp = min(perplexity, max(5.0, (n - 1) / 3))
    coords = TSNE(
        n_components=n_components,
        perplexity=perp,
        init="pca",
        random_state=42,
        learning_rate="auto",
    ).fit_transform(Z)
    return {"points": coords.tolist(), "perplexity": perp}


def project_umap(Z: np.ndarray, n_components: int = 3) -> Optional[Dict]:
    try:
        try:
            import umap
        except ImportError:
            import umap.umap_ as umap
    except ImportError:
        return None
    coords = umap.UMAP(n_components=n_components, random_state=42).fit_transform(Z)
    return {"points": coords.tolist()}


def build_model_record(path: str, methods: List[str], tsne_perplexity: float) -> Optional[Dict]:
    if not os.path.exists(path):
        print(f"Skipping missing file: {path}")
        return None

    Z, years, model_id, model_name = load_embeddings(path)
    sentences = [f"In the year {y}" for y in years]

    projections = {}
    if "pca" in methods:
        projections["pca"] = project_pca(Z)
        print(f"  [{model_name}] PCA")
    if "tsne" in methods:
        projections["tsne"] = project_tsne(Z, perplexity=tsne_perplexity)
        print(f"  [{model_name}] t-SNE")
    if "umap" in methods:
        umap_proj = project_umap(Z)
        if umap_proj is None:
            print(f"  [{model_name}] UMAP skipped (umap-learn not installed)")
        else:
            projections["umap"] = umap_proj
            print(f"  [{model_name}] UMAP")

    return {
        "id": model_id,
        "name": model_name,
        "source": os.path.basename(path),
        "years": years,
        "sentences": sentences,
        "n_points": len(years),
        "dim": int(Z.shape[1]),
        "projections": projections,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build JSON data for year embedding 3D viewer.")
    parser.add_argument(
        "--embed_paths",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "year_embeddings_viz"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pca", "tsne", "umap"],
        choices=["pca", "tsne", "umap"],
    )
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    models = []
    for path in args.embed_paths:
        print(f"Processing {path}...")
        record = build_model_record(path, args.methods, args.tsne_perplexity)
        if record is not None:
            models.append(record)

    if not models:
        raise FileNotFoundError("No embedding files were exported.")

    available_methods = sorted(
        {m for model in models for m in model["projections"].keys()}
    )
    payload = {
        "version": 1,
        "title": "Year embedding geometry",
        "subtitle": "Interactive 3D view of 'In the year x' embeddings (1700–2020)",
        "methods": available_methods,
        "models": models,
    }

    json_path = os.path.join(args.out_dir, "viz_data.json")
    js_path = os.path.join(args.out_dir, "viz_data.js")
    with open(json_path, "w") as f:
        json.dump(payload, f)
    with open(js_path, "w") as f:
        f.write("window.YEAR_EMBED_VIZ_DATA = ")
        json.dump(payload, f)
        f.write(";\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {js_path}")
    print(f"Open {os.path.join(args.out_dir, 'index.html')} in a browser.")


if __name__ == "__main__":
    main()
