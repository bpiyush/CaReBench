#!/usr/bin/env python3
"""Build interactive viz data for MatterOfTime text+image embeddings."""
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

DATASET_NAME = "time10k"

MODEL_INFO = {
    "clip": {"id": "clip", "name": "CLIP"},
    "dinotxt": {"id": "dinotxt", "name": "DINO.txt"},
    "siglip2": {"id": "siglip2", "name": "SigLIP2"},
}

CLASS_DISPLAY = {
    "cars": "Cars",
    "ships": "Ships",
    "aircrafts": "Aircraft",
    "mobilephones": "Mobile Phones",
    "musicinstruments": "Musical Instruments",
    "weapons_and_ammo": "Weapons & Ammunition",
}


def load_features(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


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


def modality_payload(feat: dict) -> dict:
    return {
        "years": [int(y) for y in feat["years"]],
        "ids": feat["ids"],
        "texts": feat.get("texts", []),
        "n_points": len(feat["ids"]),
    }


def build_class_projections(
    text_feat: dict,
    image_feat: dict | None,
    cls: str,
    methods: List[str],
    tsne_perplexity: float,
) -> Optional[Dict]:
    text_idx = [i for i, c in enumerate(text_feat["classes"]) if c == cls]
    if not text_idx:
        return None

    text_emb = text_feat["embeddings"][text_idx].numpy()
    blocks = [("text", text_emb, text_idx)]

    image_idx = []
    if image_feat is not None:
        image_idx = [i for i, c in enumerate(image_feat["classes"]) if c == cls]
        if image_idx:
            image_emb = image_feat["embeddings"][image_idx].numpy()
            blocks.append(("image", image_emb, image_idx))

    combined = np.concatenate([b[1] for b in blocks], axis=0)

    projections = {}
    for method in methods:
        if method == "pca":
            proj = project_pca(combined)
        elif method == "tsne":
            proj = project_tsne(combined, perplexity=tsne_perplexity)
        elif method == "umap":
            proj = project_umap(combined)
            if proj is None:
                continue
        else:
            continue

        points = np.asarray(proj["points"])
        offset = 0
        proj_out = {}
        for name, emb, idxs in blocks:
            n = len(idxs)
            feat = text_feat if name == "text" else image_feat
            proj_out[name] = {
                **modality_payload({k: [feat[k][i] for i in idxs] for k in ("years", "ids", "texts")}),
                "points": points[offset : offset + n].tolist(),
            }
            offset += n
        if "explained_variance_ratio" in proj:
            proj_out["explained_variance_ratio"] = proj["explained_variance_ratio"]
        if "perplexity" in proj:
            proj_out["perplexity"] = proj["perplexity"]
        projections[method] = proj_out

    return {
        "class": cls,
        "class_display": CLASS_DISPLAY.get(cls, cls),
        "n_text": len(text_idx),
        "n_image": len(image_idx),
        "has_image": len(image_idx) > 0,
        "projections": projections,
    }


def build_viz_payload(
    feat_dir: str,
    model_names: List[str],
    methods: List[str],
    tsne_perplexity: float,
) -> dict:
    models_out = []
    all_classes = set()

    for model_name in model_names:
        text_path = os.path.join(feat_dir, f"{model_name}_{DATASET_NAME}_text.pt")
        image_path = os.path.join(feat_dir, f"{model_name}_{DATASET_NAME}_image.pt")
        if not os.path.exists(text_path):
            print(f"Skipping {model_name}: missing text features")
            continue

        text_feat = load_features(text_path)
        image_feat = load_features(image_path) if os.path.exists(image_path) else None
        if image_feat is None:
            print(f"  [{model_name}] image features not found yet — text-only viz")
        classes = sorted(set(text_feat["classes"]))
        all_classes.update(classes)

        class_records = []
        for cls in classes:
            print(f"  [{model_name}] class={cls}")
            record = build_class_projections(
                text_feat, image_feat, cls, methods, tsne_perplexity
            )
            if record is not None:
                class_records.append(record)

        info = MODEL_INFO.get(model_name, {"id": model_name, "name": model_name})
        models_out.append(
            {
                **info,
                "model_id": text_feat.get("model_id", ""),
                "has_image": image_feat is not None,
                "classes": class_records,
            }
        )

    available_methods = sorted(
        {m for model in models_out for c in model["classes"] for m in c["projections"]}
    )
    return {
        "version": 1,
        "dataset": DATASET_NAME,
        "title": "MatterOfTime / TIME10k embedding explorer",
        "subtitle": "Interactive 3D PCA / t-SNE / UMAP for text and image embeddings by semantic class",
        "methods": available_methods,
        "classes": [
            {"id": c, "name": CLASS_DISPLAY.get(c, c)}
            for c in sorted(all_classes)
        ],
        "models": models_out,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feat_dir",
        type=str,
        default="/work/piyush/experiments/MatterOfTime/features",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "matteroftime_viz"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clip", "dinotxt", "siglip2"],
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

    payload = build_viz_payload(
        args.feat_dir, args.models, args.methods, args.tsne_perplexity
    )
    if not payload["models"]:
        raise FileNotFoundError(f"No feature files found in {args.feat_dir}")

    json_path = os.path.join(args.out_dir, "viz_data.json")
    js_path = os.path.join(args.out_dir, "viz_data.js")
    with open(json_path, "w") as f:
        json.dump(payload, f)
    with open(js_path, "w") as f:
        f.write("window.MATTEROFTIME_VIZ_DATA = ")
        json.dump(payload, f)
        f.write(";\n")

    print(f"\nWrote {json_path}")
    print(f"Wrote {js_path}")
    print(f"Open {os.path.join(args.out_dir, 'index.html')}")


if __name__ == "__main__":
    main()
