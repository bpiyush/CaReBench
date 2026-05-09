#!/usr/bin/env python3
"""
MSRVTT singular-value (scree) analysis for video and text embeddings:
base Tarsier2 7B vs TARA fine-tuned.

Pairs samples via data/nuanced_retrieval_data-validation-v1.csv (neg-msrvtt triplets).
Video keys: video####. Text keys: caption strings from text-standard rows.

Writes one figure with two side-by-side panels (Text | Video), using either
sigma or log(sigma) on the y-axis.

Also reports Wang & Isola (ICML 2020) hyperspherical uniformity
L_uniform = log E[exp(-t ||f(x)-f(x')||^2)] on row L2-normalized embeddings
(lower => more uniform on the sphere; paper default t=2).

Example:
  python evals_tarsier2/analyze_msrvtt_svd_base_vs_tara.py \\
    --out_dir outputs/msrvtt_svd_tarsier

Default output is a compact PDF (text panel left, video right).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

VIDEO_KEY_RE = re.compile(r"^video\d+$")


def _load_emb_dict(path: str) -> Dict[str, torch.Tensor]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Embeddings file not found: {path}")
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    return obj


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
        if not VIDEO_KEY_RE.match(vid):
            continue
        pairs.append((vid, tid))
    return pairs


def stack_embeddings(
    emb: Dict[str, torch.Tensor], keys: List[str], label: str
) -> np.ndarray:
    missing = [k for k in keys if k not in emb]
    if missing:
        raise KeyError(
            f"{label}: missing {len(missing)} keys (showing up to 5): {missing[:5]}"
        )
    return np.stack([emb[k].float().numpy() for k in keys], axis=0).astype(np.float64)


def centered_singular_values(X: np.ndarray) -> np.ndarray:
    """X: (N, D). Return singular values in descending order."""
    Xc = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    return s


def svd_metrics(s: np.ndarray, embedding_dim: int) -> dict:
    """embedding_dim D is feature size (used for global participation ratio)."""
    energy = s**2
    total = float(np.sum(energy)) + 1e-12
    sum_energy_sq = float(np.sum(energy**2)) + 1e-12
    cum = np.cumsum(energy) / total
    p = energy / total
    ent = float(-np.sum(p * np.log(p + 1e-12)))
    # Geometric metrics on covariance eigenvalues λ_i ∝ s_i^2
    global_isotropy_score = float(1.0 - energy[0] / total)
    global_participation_ratio = float((total**2) / (embedding_dim * sum_energy_sq))
    return {
        "num_singular_values": int(len(s)),
        "mean_singular_value": float(np.mean(s)),
        "std_singular_value": float(np.std(s)),
        "max_singular_value": float(np.max(s)),
        "effective_rank": float(np.exp(ent)),
        "top1_energy_ratio": float(energy[0] / total),
        "global_isotropy_score": global_isotropy_score,
        "global_participation_ratio": global_participation_ratio,
        "num_sv_for_90pct_energy": int(np.searchsorted(cum, 0.90) + 1),
        "num_sv_for_95pct_energy": int(np.searchsorted(cum, 0.95) + 1),
    }


def wang_isola_uniformity_loss(X: np.ndarray, t: float = 2.0) -> float:
    """Wang & Isola (2020): L_uniform = log E[exp(-t ||f(x)-f(x')||^2)].

    f rows are L2-normalized to the unit hypersphere. Expectation is over all
    ordered pairs (x, x') with x != x', matching torch.pdist-style pair means.
    Lower values indicate more spatially uniform distributions on the sphere.
    """
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    F = X / np.maximum(norms, 1e-12)
    gram = F @ F.T
    d2 = 2.0 - 2.0 * gram
    np.fill_diagonal(d2, 0.0)
    n = int(F.shape[0])
    if n < 2:
        return float("nan")
    w = np.exp(-t * d2)
    np.fill_diagonal(w, 0.0)
    mean_pair = float(w.sum() / (n * (n - 1)))
    return float(np.log(mean_pair + 1e-12))


def plot_two_panel_screes(
    out_path: str,
    sv_video_base: np.ndarray,
    sv_video_tara: np.ndarray,
    sv_text_base: np.ndarray,
    sv_text_tara: np.ndarray,
    use_log: bool,
    font_scale: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_sizes = {
        "font.size": 14.0,
        "axes.titlesize": 16.0,
        "axes.labelsize": 15.0,
        "xtick.labelsize": 12.0,
        "ytick.labelsize": 12.0,
        "legend.fontsize": 12.0,
    }
    plt.rcParams.update(
        {
            "font.family": "serif",
            **{k: v * font_scale for k, v in base_sizes.items()},
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), sharex=True, sharey=False)

    def draw_compare(
        ax,
        sv_base: np.ndarray,
        sv_tara: np.ndarray,
        title: str,
    ) -> None:
        mb = float(np.mean(sv_base))
        mt = float(np.mean(sv_tara))
        xb = np.arange(1, len(sv_base) + 1)
        xt = np.arange(1, len(sv_tara) + 1)
        if use_log:
            eps = np.finfo(float).tiny
            sb = np.maximum(sv_base, eps)
            st = np.maximum(sv_tara, eps)
            yb = np.log(sb)
            yt = np.log(st)
            ax.plot(xb, yb, color="C0", lw=2, label=f"Base (Tarsier 2 7B), μ={mb:.3f}")
            ax.plot(xt, yt, color="C1", lw=2, label=f"TARA (Ours), μ={mt:.3f}")
            ax.set_ylim(-5, 5)
            ax.set_ylabel(r"$\log(\sigma)$")
            ax.grid(alpha=0.25)
        else:
            ax.plot(xb, sv_base, color="C0", lw=2, label=f"Base (Tarsier 2 7B), μ={mb:.3f}")
            ax.plot(xt, sv_tara, color="C1", lw=2, label=f"TARA (Ours), μ={mt:.3f}")
            ax.set_ylabel(r"$\sigma$")
            ax.grid(alpha=0.25)

        ax.set_title(title)
        ax.set_xlabel("Index")
        ax.legend(loc="upper right", frameon=False)

    draw_compare(axes[0], sv_text_base, sv_text_tara, "Text")
    draw_compare(axes[1], sv_video_base, sv_video_tara, "Video")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fmt = "pdf" if out_path.lower().endswith(".pdf") else None
    plt.savefig(out_path, format=fmt, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SVD scree plots: MSRVTT video & text, base vs TARA."
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="./data/nuanced_retrieval_data-validation-v1.csv",
        help="CSV with neg-msrvtt triplets (video / text-standard / text-negation).",
    )
    parser.add_argument(
        "--base_video_embeddings",
        type=str,
        default="/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/embs/"
        "tarsier2_7b_nuanced_retrieval_embeddings.pt",
        help="Base model video (and optionally other) embeddings.",
    )
    parser.add_argument(
        "--base_text_embeddings",
        type=str,
        default="/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/embs/"
        "tarsier2_7b_nuanced_retrieval_data-validation-v1_embeddings.pt",
        help="Base model embeddings that include MSRVTT text-standard keys.",
    )
    parser.add_argument(
        "--tara_video_embeddings",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier2-7b-0115/special_milestones/"
        "Tarsier2-TARA-chiral10k_covr10k/embs/"
        "tarsier2+tara_nuanced_retrieval_data-v1_embeddings.pt",
        help="TARA fine-tuned embeddings (must contain same video#### keys).",
    )
    parser.add_argument(
        "--tara_text_embeddings",
        type=str,
        default=None,
        help="TARA embeddings with same text keys as CSV. "
        "Default: try ..._data-validation-v1_embeddings.pt next to the video "
        "file; if missing, reuse --tara_video_embeddings (TARA v1 often "
        "contains both modalities).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/msrvtt_svd_base_vs_tara",
        help="Directory for plot and metrics JSON.",
    )
    parser.add_argument(
        "--plot_name",
        type=str,
        default="msrvtt_svd_scree_base_vs_tara.pdf",
        help="Output plot filename under out_dir (PDF recommended).",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="If set, plot log(sigma) on the y-axis. Default plots sigma directly.",
    )
    parser.add_argument(
        "--font_scale",
        type=float,
        default=1.0,
        help="Global multiplier for all plot font sizes.",
    )
    parser.add_argument(
        "--uniformity_t",
        type=float,
        default=2.0,
        help="Temperature t in Wang & Isola L_uniform (paper uses t=2).",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(csv_path)

    pairs = msrvtt_video_text_pairs(csv_path)
    if not pairs:
        raise RuntimeError(f"No MSRVTT pairs parsed from {csv_path}")

    video_keys = [p[0] for p in pairs]
    text_keys = [p[1] for p in pairs]

    emb_bv = _load_emb_dict(args.base_video_embeddings)
    emb_bt = _load_emb_dict(args.base_text_embeddings)
    emb_tv = _load_emb_dict(args.tara_video_embeddings)

    tara_text_path = args.tara_text_embeddings
    if tara_text_path is None:
        d = os.path.dirname(os.path.abspath(args.tara_video_embeddings))
        candidate = os.path.join(
            d, "tarsier2+tara_nuanced_retrieval_data-validation-v1_embeddings.pt"
        )
        if os.path.isfile(candidate):
            tara_text_path = candidate
        else:
            tara_text_path = args.tara_video_embeddings
    emb_tt = _load_emb_dict(tara_text_path)

    # Intersect video keys present in both base and TARA video dicts
    vb_ok = [k for k in video_keys if k in emb_bv and k in emb_tv]
    if len(vb_ok) < len(video_keys):
        print(
            f"Warning: using {len(vb_ok)}/{len(video_keys)} videos "
            f"present in both base and TARA video files."
        )
    video_keys_use = vb_ok

    tb_ok = [k for k in text_keys if k in emb_bt and k in emb_tt]
    if len(tb_ok) < len(text_keys):
        print(
            f"Warning: using {len(tb_ok)}/{len(text_keys)} text captions "
            f"present in both base and TARA text files."
        )
    text_keys_use = tb_ok

    if not video_keys_use:
        raise RuntimeError("No overlapping MSRVTT video keys between base and TARA.")
    if not text_keys_use:
        raise RuntimeError(
            "No overlapping MSRVTT text keys. "
            "Compute TARA validation embeddings with the same CSV, or pass "
            "--tara_text_embeddings explicitly."
        )

    # Align rows: use intersection of indices where both modalities exist
    vid_set = set(video_keys_use)
    txt_set = set(text_keys_use)
    aligned = [(v, t) for v, t in pairs if v in vid_set and t in txt_set]
    if not aligned:
        raise RuntimeError("No aligned (video, text) pairs after key filtering.")

    vk = [a[0] for a in aligned]
    tk = [a[1] for a in aligned]

    X_vb = stack_embeddings(emb_bv, vk, "base video")
    X_tv = stack_embeddings(emb_tv, vk, "TARA video")
    X_tb = stack_embeddings(emb_bt, tk, "base text")
    X_tt = stack_embeddings(emb_tt, tk, "TARA text")

    sv_vb = centered_singular_values(X_vb)
    sv_tv = centered_singular_values(X_tv)
    sv_tb = centered_singular_values(X_tb)
    sv_tt = centered_singular_values(X_tt)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, args.plot_name)
    plot_two_panel_screes(
        plot_path,
        sv_vb,
        sv_tv,
        sv_tb,
        sv_tt,
        use_log=args.log,
        font_scale=args.font_scale,
    )

    D = int(X_vb.shape[1])
    t_unif = float(args.uniformity_t)
    L_vid_b = wang_isola_uniformity_loss(X_vb, t_unif)
    L_vid_t = wang_isola_uniformity_loss(X_tv, t_unif)
    L_txt_b = wang_isola_uniformity_loss(X_tb, t_unif)
    L_txt_t = wang_isola_uniformity_loss(X_tt, t_unif)
    report = {
        "csv_path": csv_path,
        "n_samples_aligned": len(aligned),
        "embedding_dim": D,
        "paths": {
            "base_video": os.path.abspath(args.base_video_embeddings),
            "base_text": os.path.abspath(args.base_text_embeddings),
            "tara_video": os.path.abspath(args.tara_video_embeddings),
            "tara_text": os.path.abspath(tara_text_path),
        },
        "video": {
            "base": svd_metrics(sv_vb, D),
            "tara": svd_metrics(sv_tv, D),
        },
        "text_standard": {
            "base": svd_metrics(sv_tb, D),
            "tara": svd_metrics(sv_tt, D),
        },
        "wang_isola_uniformity": {
            "reference": "Wang & Isola, Understanding Contrastive Representation Learning "
            "through Alignment and Uniformity on the Hypersphere (ICML 2020)",
            "t": t_unif,
            "L_uniform": {
                "video": {"base": L_vid_b, "tara": L_vid_t},
                "text_standard": {"base": L_txt_b, "tara": L_txt_t},
            },
        },
        "plot": plot_path,
    }
    json_path = os.path.join(out_dir, "msrvtt_svd_metrics_base_vs_tara.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Wrote plot:  {plot_path}")
    print(f"Wrote metrics: {json_path}")
    print(f"Aligned samples: {len(aligned)}, D={D}")
    gi_txt_b = report["text_standard"]["base"]["global_isotropy_score"]
    gi_txt_t = report["text_standard"]["tara"]["global_isotropy_score"]
    gi_vid_b = report["video"]["base"]["global_isotropy_score"]
    gi_vid_t = report["video"]["tara"]["global_isotropy_score"]
    print(
        "Global isotropy score G.Iso = 1 - s_1^2 / sum_i s_i^2 "
        "(same as 1 - lambda_1 / sum_i lambda_i for lambda_i propto s_i^2):"
    )
    print(f"  Text:  base (Tarsier2) = {gi_txt_b:.6f},  TARA (fine-tuned) = {gi_txt_t:.6f}")
    print(f"  Video: base (Tarsier2) = {gi_vid_b:.6f},  TARA (fine-tuned) = {gi_vid_t:.6f}")
    gpr_txt_b = report["text_standard"]["base"]["global_participation_ratio"]
    gpr_txt_t = report["text_standard"]["tara"]["global_participation_ratio"]
    gpr_vid_b = report["video"]["base"]["global_participation_ratio"]
    gpr_vid_t = report["video"]["tara"]["global_participation_ratio"]
    print(
        "Global participation ratio G.PR = (sum_i lambda_i)^2 / (D sum_i lambda_i^2) "
        "= (sum_i s_i^2)^2 / (D sum_i s_i^4):"
    )
    print(f"  Text:  base = {gpr_txt_b:.6f},  TARA = {gpr_txt_t:.6f}")
    print(f"  Video: base = {gpr_vid_b:.6f},  TARA = {gpr_vid_t:.6f}")
    print(
        f"Wang & Isola uniformity L_uniform = log E[exp(-t||f(x)-f(x')||^2)], "
        f"t={t_unif}, f = row L2-normalized (lower => more uniform):"
    )
    print(f"  Text:  base (Tarsier2) = {L_txt_b:.6f},  TARA (fine-tuned) = {L_txt_t:.6f}")
    print(f"  Video: base (Tarsier2) = {L_vid_b:.6f},  TARA (fine-tuned) = {L_vid_t:.6f}")


if __name__ == "__main__":
    main()
