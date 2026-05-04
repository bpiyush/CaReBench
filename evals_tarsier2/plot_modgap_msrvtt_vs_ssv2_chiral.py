#!/usr/bin/env python3
"""
Scatter plot: MSRVTT modality gap (x) vs retrieval metric(s) (y) across ablations.

- x: Same definition as evals_tarsier2/measure_modgap_over_training.py:
     L2 norm of (mean normalized video embeddings - mean normalized text embeddings)
     over aligned MSRVTT pairs from the nuanced validation CSV.
- y (default ``--y_metric composite``): mean of SSv2 time-t2v chiral R@1, WebVid CoVR R@1
  (``multimodal_covr.covr``), and MSRVTT negation-track R@5 (``negation-msrvtt`` branch).

Expects parallel line-aligned lists:
  metric_files.txt  — one JSON path per line
  embedding_files.txt — one .pt path per line (same order / one experiment per row)

Example:
  python evals_tarsier2/plot_modgap_msrvtt_vs_ssv2_chiral.py \\
    --metric_files_txt /work/piyush/experiments/CaRe/Tarsier2-7b-0115/ablations/metric_files.txt \\
    --embedding_files_txt /work/piyush/experiments/CaRe/Tarsier2-7b-0115/ablations/embedding_files.txt \\
    --csv_path data/nuanced_retrieval_data-validation-v1.csv \\
    --out_plot outputs/modgap_vs_ssv2_chiral.pdf
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

try:
    from sklearn.linear_model import LinearRegression, RANSACRegressor
except ImportError:  # pragma: no cover
    RANSACRegressor = None  # type: ignore[misc, assignment]
    LinearRegression = None  # type: ignore[misc, assignment]

# Triplet pairing aligned with evals_tarsier2/analyze_msrvtt_svd_base_vs_tara.py
_VIDEO_KEY_RE = re.compile(r"^video\d+$")


def msrvtt_video_text_pairs(csv_path: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(csv_path)
    ms = df[df["source"] == "neg-msrvtt"].reset_index(drop=True)
    pairs: List[Tuple[str, str]] = []
    for i in range(0, len(ms), 3):
        chunk = ms.iloc[i : i + 3]
        if len(chunk) < 3:
            continue
        if set(chunk["modality"]) != {"video", "text-standard", "text-negation"}:
            continue
        vid = str(chunk[chunk["modality"] == "video"].iloc[0]["id"])
        tid = str(chunk[chunk["modality"] == "text-standard"].iloc[0]["id"])
        if not _VIDEO_KEY_RE.match(vid):
            continue
        pairs.append((vid, tid))
    return pairs


def _read_nonempty_lines(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def _resolve_list_path(raw: str, root: str) -> str:
    """Absolute paths unchanged; relative paths joined to root (ablations dir)."""
    raw = os.path.expanduser(raw.strip())
    if os.path.isabs(raw):
        return os.path.normpath(raw)
    if not root:
        return os.path.normpath(raw)
    return os.path.normpath(os.path.join(root, raw))


def modality_gap_from_stacks(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Match measure_modgap_over_training.modality_gap_from_tensors (without np.round)."""
    return float((X.mean(dim=0) - Y.mean(dim=0)).norm(dim=-1).item())


def compute_msrvtt_modality_gap(
    embs: Dict[str, torch.Tensor], pairs: List[Tuple[str, str]]
) -> float:
    xs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    for vid, tid in pairs:
        if vid not in embs or tid not in embs:
            continue
        xs.append(embs[vid].flatten())
        ys.append(embs[tid].flatten())
    if not xs:
        raise RuntimeError(
            "No aligned MSRVTT (video, text-standard) pairs found in embedding dict."
        )
    X = torch.nn.functional.normalize(torch.stack(xs), dim=-1)
    Y = torch.nn.functional.normalize(torch.stack(ys), dim=-1)
    return modality_gap_from_stacks(X, Y)


def _chiral_block(metrics: Dict[str, Any]) -> Dict[str, Any] | None:
    for key in ("time_t2v-ssv2", "time_t2v"):
        block = metrics.get(key)
        if not isinstance(block, dict):
            continue
        ch = block.get("chiral")
        if isinstance(ch, dict) and ch:
            return ch
    return None


def _ssv2_chiral_r1(metrics: Dict[str, Any]) -> float:
    ch = _chiral_block(metrics)
    if ch is None:
        raise KeyError(
            "Could not find SSv2 chiral metrics (tried time_t2v-ssv2/chiral "
            "and time_t2v/chiral)."
        )
    v = ch.get("R@1")
    if v is None:
        raise KeyError("chiral block has no R@1")
    return float(v)


def _ssv2_chiral_mean_r1_r5_r10(metrics: Dict[str, Any]) -> float:
    ch = _chiral_block(metrics)
    if ch is None:
        raise KeyError("Could not find SSv2 chiral metrics.")
    vals = []
    for k in ("R@1", "R@5", "R@10"):
        if ch.get(k) is not None:
            vals.append(float(ch[k]))
    if not vals:
        raise KeyError("chiral block missing R@1, R@5, R@10")
    return float(np.mean(vals))


def _covr_r1(metrics: Dict[str, Any]) -> float:
    mm = metrics.get("multimodal_covr")
    if not isinstance(mm, dict):
        raise KeyError("multimodal_covr missing")
    covr = mm.get("covr")
    if not isinstance(covr, dict):
        raise KeyError("multimodal_covr.covr missing")
    v = covr.get("R@1")
    if v is None:
        raise KeyError("CoVR R@1 missing")
    return float(v)


def _msrvtt_negation_track_r5(metrics: Dict[str, Any]) -> float:
    for key in ("negation-msrvtt", "negation_msrvtt"):
        block = metrics.get(key)
        if not isinstance(block, dict):
            continue
        neg = block.get("negation")
        if not isinstance(neg, dict):
            continue
        inner = neg.get("negation")
        if not isinstance(inner, dict):
            continue
        v = inner.get("R@5")
        if v is not None:
            return float(v)
    raise KeyError(
        "MSRVTT negation-track R@5 not found (negation-msrvtt/negation/negation/R@5)."
    )


def y_from_metrics(metrics: Dict[str, Any], y_metric: str) -> float:
    if y_metric == "composite":
        a = _ssv2_chiral_r1(metrics)
        b = _covr_r1(metrics)
        c = _msrvtt_negation_track_r5(metrics)
        return float(np.mean([a, b, c]))
    if y_metric == "ssv2_chiral_r1":
        return _ssv2_chiral_r1(metrics)
    if y_metric == "ssv2_chiral_mean_r1_r5_r10":
        return _ssv2_chiral_mean_r1_r5_r10(metrics)
    raise ValueError(f"Unknown --y_metric: {y_metric}")


def experiment_label(metric_path: str, emb_path: str, index: int) -> str:
    # .../core.N-time.../merged_checkpoint/metrics/metrics_*.json
    m = os.path.normpath(metric_path)
    exp_dir = os.path.dirname(os.path.dirname(os.path.dirname(m)))
    base = os.path.basename(exp_dir)
    if base.startswith("core."):
        return base
    stem = os.path.splitext(os.path.basename(emb_path))[0]
    return stem or f"exp_{index}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot MSRVTT modality gap vs retrieval metric(s) for listed ablations."
    )
    p.add_argument(
        "--metric_files_txt",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier2-7b-0115/ablations/metric_files.txt",
        help="Text file: one metrics JSON path per line.",
    )
    p.add_argument(
        "--embedding_files_txt",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier2-7b-0115/ablations/embedding_files.txt",
        help="Text file: one embedding .pt path per line (same order as metrics).",
    )
    p.add_argument(
        "--csv_path",
        type=str,
        default="data/nuanced_retrieval_data-validation-v1.csv",
        help="Nuanced retrieval CSV (MSRVTT triplets for pairing).",
    )
    p.add_argument(
        "--out_plot",
        type=str,
        default="outputs/modgap_msrvtt_vs_ssv2_chiral.pdf",
        help="Output figure path.",
    )
    p.add_argument(
        "--y_metric",
        type=str,
        choices=("composite", "ssv2_chiral_r1", "ssv2_chiral_mean_r1_r5_r10"),
        default="composite",
        help="Y-axis: default = mean(SSv2 chiral R@1 t2v, CoVR R@1, MSRVTT negation R@5); "
        "or SSv2-only variants.",
    )
    p.add_argument(
        "--annotate",
        action="store_true",
        help="If set, label each point with a short name derived from paths.",
    )
    p.add_argument(
        "--paths_root",
        type=str,
        default="",
        help="Directory to prepend to relative paths in the two list files. "
        "Default: directory containing --metric_files_txt.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metric_paths = _read_nonempty_lines(args.metric_files_txt)
    emb_paths = _read_nonempty_lines(args.embedding_files_txt)
    if len(metric_paths) != len(emb_paths):
        raise ValueError(
            f"Line count mismatch: {len(metric_paths)} metric paths vs "
            f"{len(emb_paths)} embedding paths."
        )

    paths_root = args.paths_root or os.path.dirname(
        os.path.abspath(args.metric_files_txt)
    )

    if not os.path.isfile(args.csv_path):
        raise FileNotFoundError(f"CSV not found: {args.csv_path}")
    pairs = msrvtt_video_text_pairs(args.csv_path)
    if not pairs:
        raise RuntimeError(f"No MSRVTT pairs parsed from {args.csv_path}")

    gaps: List[float] = []
    ys: List[float] = []
    labels: List[str] = []
    used_paths: List[Tuple[str, str]] = []

    for i, (m_raw, e_raw) in enumerate(zip(metric_paths, emb_paths)):
        m_path = _resolve_list_path(m_raw, paths_root)
        e_path = _resolve_list_path(e_raw, paths_root)
        if not os.path.isfile(m_path):
            print(f"[skip] missing metrics: {m_path}")
            continue
        if not os.path.isfile(e_path):
            print(f"[skip] missing embeddings: {e_path}")
            continue
        with open(m_path, encoding="utf-8") as f:
            metrics = json.load(f)
        try:
            y_val = y_from_metrics(metrics, args.y_metric)
        except KeyError as e:
            print(f"[skip] {m_path}: {e}")
            continue

        embs = torch.load(e_path, map_location="cpu")
        if not isinstance(embs, dict):
            print(f"[skip] expected dict in {e_path}, got {type(embs)}")
            continue
        embs_str = {str(k): v for k, v in embs.items()}
        try:
            gap = compute_msrvtt_modality_gap(embs_str, pairs)
        except RuntimeError as e:
            print(f"[skip] {e_path}: {e}")
            continue

        gaps.append(gap)
        ys.append(y_val)
        labels.append(experiment_label(m_path, e_path, i))
        used_paths.append((m_path, e_path))

    if len(gaps) < 2:
        raise RuntimeError(
            f"Need at least 2 valid points for a useful plot; got {len(gaps)}."
        )

    x = np.array(gaps, dtype=np.float64)
    y = np.array(ys, dtype=np.float64)

    if RANSACRegressor is None:
        raise ImportError(
            "RANSAC fit requires scikit-learn. Install with: pip install scikit-learn"
        )
    X = x.reshape(-1, 1)
    ransac = RANSACRegressor(
        estimator=LinearRegression(),
        min_samples=2,
        random_state=0,
    )
    ransac.fit(X, y)
    coef = np.asarray(ransac.estimator_.coef_).reshape(-1)
    slope = float(coef[0]) if coef.size else 0.0
    intercept = float(ransac.estimator_.intercept_)
    x_line = np.linspace(float(x.min()), float(x.max()), 200)
    y_line = slope * x_line + intercept

    inlier_mask = np.asarray(ransac.inlier_mask_, dtype=bool)
    x_in = x[inlier_mask]
    y_in = y[inlier_mask]
    rho_inliers = float("nan")
    if x_in.size >= 2 and np.std(x_in) > 1e-12 and np.std(y_in) > 1e-12:
        try:
            from scipy.stats import pearsonr

            rho_inliers, _ = pearsonr(x_in, y_in)
            rho_inliers = float(rho_inliers)
        except Exception:
            rho_inliers = float(np.corrcoef(x_in, y_in)[0, 1])

    line_label = rf"RANSAC: $y = {slope:.2f}\,x + {intercept:.2f}$"
    if not np.isnan(rho_inliers):
        line_label += rf", $\rho={rho_inliers:.2f}$ (inliers)"

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(
        x,
        y,
        s=42,
        alpha=0.85,
        edgecolors="k",
        linewidths=0.4,
        zorder=3,
        label="Ablations",
    )
    ax.plot(
        x_line,
        y_line,
        color="tab:red",
        lw=2.2,
        zorder=2,
        label=line_label,
    )

    if args.annotate:
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(
                lab,
                (xi, yi),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=6,
                alpha=0.9,
            )

    ax.set_xlabel("MSRVTT modality gap $||\\Delta_{gap}||_{2}$ (nuanced CSV pairs)")
    if args.y_metric == "composite":
        y_label = "Mean: SSv2-chiral R@1, CoVR R@1, MSR neg. R@5"
    elif args.y_metric == "ssv2_chiral_r1":
        y_label = "SSv2 chiral R@1 (time t2v)"
    else:
        y_label = r"SSv2 chiral mean(R@1,R@5,R@10) (time t2v)"
    ax.set_ylabel(y_label)
    title = (
        "Modality gap vs. nuanced retrieval"
        if args.y_metric == "composite"
        else "Modality gap vs. SSv2 chiral performance"
    )
    ax.set_title(title)
    ax.grid(alpha=0.25, zorder=0)
    ax.legend(loc="lower left", fontsize=11)

    out = args.out_plot
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out} ({len(gaps)} points).")
    print(
        f"RANSAC line: slope={slope:.4f}, intercept={intercept:.4f}, "
        f"inliers={int(inlier_mask.sum())}/{len(x)}, rho_inliers={rho_inliers:.4f}"
    )
    for lab, g, yy, (mp, ep) in zip(labels, gaps, ys, used_paths):
        print(f"  {lab:40s}  gap={g:.4f}  y={yy:.3f}")
    print("Saved plot to ", out)


if __name__ == "__main__":
    main()
