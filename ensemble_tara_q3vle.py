"""
Ensemble retrieval metrics: s(q,c) = alpha * s_TARA(q,c) + (1 - alpha) * s_Qwen(q,c).

Run from the CaReBench repo root so ./data/ and shared resolve correctly:
  python ensemble_tara_q3vle.py --alpha 0.5
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import shared.utils as su  # noqa: E402
from evals_tarsier2.compute_metrics_validation import (  # noqa: E402
    compute_metrics_multimodal_covr,
    compute_metrics_negation,
    compute_metrics_time_t2v,
    compute_metrics_time_v2t,
)


def main():
    parser = argparse.ArgumentParser(description="TARA + Qwen3-VL-E ensemble retrieval metrics")
    parser.add_argument(
        "--tara_feat_path",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k/merged_checkpoint/embs/tarsier2+tara_nuanced_retrieval_data-v1_embeddings.pt",
    )
    parser.add_argument(
        "--qwen_feat_path",
        type=str,
        default="/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B/embs/qwen3vlemb_nuanced_retrieval_data-v1_embeddings.pt",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Weight on TARA similarity; Qwen gets (1 - alpha).",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./data/nuanced_retrieval_data-v1.csv",
    )
    parser.add_argument(
        "--lab_path",
        type=str,
        default="./data/nuanced_retrieval_labels.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for metrics JSON (default: ./_ensemble_metrics).",
    )
    args = parser.parse_args()

    assert 0.0 <= args.alpha <= 1.0, "alpha must be in [0, 1]"

    for path, name in [
        (args.tara_feat_path, "TARA"),
        (args.qwen_feat_path, "Qwen3-VL-E"),
        (args.csv_path, "CSV"),
        (args.lab_path, "labels JSON"),
    ]:
        assert os.path.exists(path), f"{name} file does not exist: {path}"

    print(f"Loading TARA embeddings from {args.tara_feat_path}")
    feat_tara = torch.load(args.tara_feat_path, weights_only=False)
    print(f"  keys: {len(feat_tara)}")

    print(f"Loading Qwen3-VL-E embeddings from {args.qwen_feat_path}")
    feat_qwen = torch.load(args.qwen_feat_path, weights_only=False)
    print(f"  keys: {len(feat_qwen)}")

    keys_tara = set(feat_tara.keys())
    keys_qwen = set(feat_qwen.keys())
    if keys_tara != keys_qwen:
        only_t = keys_tara - keys_qwen
        only_q = keys_qwen - keys_tara
        print(
            "Warning: embedding key sets differ — "
            f"only in TARA: {len(only_t)}, only in Qwen: {len(only_q)}. "
            "Metrics skip queries/candidates missing from either dict."
        )

    df = pd.read_csv(args.csv_path)
    labels = su.io.load_json(args.lab_path)
    csv_name = os.path.basename(args.csv_path).rsplit(".", 1)[0]
    print(f"Loaded {len(df)} rows, {len(labels)} label entries.")

    metrics = {}

    for dataset in ["ssv2", "epic", "charades"]:
        metrics[f"time_t2v-{dataset}"] = compute_metrics_time_t2v(
            df, feat_tara, labels, dataset, feat_b=feat_qwen, alpha=args.alpha
        )
        metrics[f"time_v2t-{dataset}"] = compute_metrics_time_v2t(
            df, feat_tara, labels, dataset, feat_b=feat_qwen, alpha=args.alpha
        )

    for dataset in ["coco", "msrvtt"]:
        metrics[f"negation-{dataset}"] = compute_metrics_negation(
            df, feat_tara, labels, dataset, feat_b=feat_qwen, alpha=args.alpha
        )

    metrics["multimodal_covr"] = compute_metrics_multimodal_covr(
        df, feat_tara, labels, feat_b=feat_qwen, alpha=args.alpha
    )

    out_dir = args.output_dir or os.path.join(_ROOT, "_ensemble_metrics")
    os.makedirs(out_dir, exist_ok=True)
    tag = f"ensemble_tara_q3vle_a{args.alpha:g}_{csv_name}".replace(".", "p")
    save_path = os.path.join(out_dir, f"metrics_{tag}.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {save_path}")
    
    # Print metrics
    lines = []
    X = metrics
    for ds in ['ssv2', 'epic', 'charades']:
        lines.append(f"time_v2t-{ds} & {X[f'time_v2t-{ds}']['chiral']['R@1']} & {X[f'time_v2t-{ds}']['static']['R@1']} & {X[f'time_v2t-{ds}']['all']['R@1']}")
    for ds in ['coco', 'msrvtt']:
        lines.append(f"negation-{ds} & {X[f'negation-{ds}']['standard']['standard']['R@1']} & {X[f'negation-{ds}']['negation']['negation']['R@1']}")
    lines.append(f"multimodal_covr & {X['multimodal_covr']['covr']['R@1']} & {X['multimodal_covr']['covr']['R@5']}")
    print("\n & ".join(lines))


if __name__ == "__main__":
    main()
