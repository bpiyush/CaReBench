"""
Estimate Modality Gap Assumption (MGA) terms on MSRVTT test-1K embeddings.

Assumption: for paired (v, t),  v - t = c_perp + eps
  - c_perp constant and orthogonal to video/text embedding spans
  - eps ~ N(0, nu^2 I)

Estimators:
  c_hat = mean_i (v_i - t_i)
  eps_i = (v_i - t_i) - c_hat
  Orthogonality: cosine of c_hat with top-k PCA directions of V and T,
                 and mean |<c_hat, v_i>| / (|c_hat||v_i|), same for t.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--emb_paths",
        nargs="+",
        required=True,
        help="One or more *.pt embedding dicts (video_id + caption keys).",
    )
    p.add_argument(
        "--data_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/MSRVTT",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/mga_msrvtt_test1k",
    )
    p.add_argument("--topk", type=int, default=10, help="PCA dirs for orthogonality check.")
    return p.parse_args()


def load_aligned(emb_path: str, df: pd.DataFrame):
    embs = torch.load(emb_path, map_location="cpu", weights_only=False)
    V, T, ids = [], [], []
    for _, row in df.iterrows():
        vid = str(row["video_id"])
        cap = str(row["caption"])
        if vid in embs and cap in embs:
            V.append(embs[vid].float())
            T.append(embs[cap].float())
            ids.append(vid)
    if not V:
        raise RuntimeError(f"No aligned pairs in {emb_path}")
    V = torch.nn.functional.normalize(torch.stack(V), dim=-1)
    T = torch.nn.functional.normalize(torch.stack(T), dim=-1)
    return V, T, ids


def pca_basis(X: torch.Tensor, k: int):
    # X: [N, D], centered
    Xc = X - X.mean(dim=0, keepdim=True)
    # economy SVD via torch
    # use numpy for numerical stability on CPU float64
    U, S, Vt = np.linalg.svd(Xc.double().numpy(), full_matrices=False)
    k = min(k, Vt.shape[0])
    return torch.from_numpy(Vt[:k]).float(), S[:k]


def analyze(name: str, V: torch.Tensor, T: torch.Tensor, topk: int):
    diffs = V - T  # [N, D]
    c = diffs.mean(dim=0)
    eps = diffs - c
    c_norm = c.norm().item()
    eps_norms = eps.norm(dim=-1)
    # isotropic noise estimate: mean per-dim variance
    nu2 = eps.var(dim=0, unbiased=True).mean().item()
    nu = float(np.sqrt(max(nu2, 0.0)))

    # fraction of ||v-t||^2 explained by c vs residual
    diff_sq = (diffs ** 2).sum(dim=-1)
    c_sq = c_norm ** 2
    eps_sq = (eps ** 2).sum(dim=-1)
    explained = float((c_sq / (diff_sq.mean().item() + 1e-12)))

    # orthogonality vs sample means / individual vectors
    def mean_abs_cos(a, X):
        a_n = a / (a.norm() + 1e-12)
        Xn = torch.nn.functional.normalize(X, dim=-1)
        return (Xn @ a_n).abs().mean().item()

    cos_v_mean = mean_abs_cos(c, V.mean(dim=0, keepdim=True))
    cos_t_mean = mean_abs_cos(c, T.mean(dim=0, keepdim=True))
    cos_v = mean_abs_cos(c, V)
    cos_t = mean_abs_cos(c, T)

    # PCA subspace angles
    Bv, Sv = pca_basis(V, topk)
    Bt, St = pca_basis(T, topk)
    c_u = c / (c.norm() + 1e-12)
    cos_pca_v = (Bv @ c_u).abs()
    cos_pca_t = (Bt @ c_u).abs()

    # residual isotropy: ratio of max/min per-dim variance
    per_dim_var = eps.var(dim=0, unbiased=True)
    iso_ratio = (per_dim_var.max() / (per_dim_var.min() + 1e-12)).item()

    # constancy of gap: relative std of (v-t) projected onto c direction
    proj = diffs @ c_u
    proj_std = proj.std(unbiased=True).item()
    proj_mean = proj.mean().item()

    out = {
        "name": name,
        "n_pairs": int(V.shape[0]),
        "dim": int(V.shape[1]),
        "||c_perp||": c_norm,
        "mean||eps||": eps_norms.mean().item(),
        "std||eps||": eps_norms.std(unbiased=True).item(),
        "nu (sqrt mean per-dim var)": nu,
        "frac_energy_in_c": explained,
        "mean_abs_cos(c, v_i)": cos_v,
        "mean_abs_cos(c, t_i)": cos_t,
        "abs_cos(c, mean_v)": cos_v_mean,
        "abs_cos(c, mean_t)": cos_t_mean,
        f"max_abs_cos(c, PCA_v[:{topk}])": cos_pca_v.max().item(),
        f"mean_abs_cos(c, PCA_v[:{topk}])": cos_pca_v.mean().item(),
        f"max_abs_cos(c, PCA_t[:{topk}])": cos_pca_t.max().item(),
        f"mean_abs_cos(c, PCA_t[:{topk}])": cos_pca_t.mean().item(),
        "gap_proj_mean": proj_mean,
        "gap_proj_std": proj_std,
        "eps_isotropy_max/min_var": iso_ratio,
        "top5_video_pca_singvals": Sv[:5].tolist(),
        "top5_text_pca_singvals": St[:5].tolist(),
    }
    return out, c, eps


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        json.load(open(os.path.join(args.data_root, "annotation", "msrvtt_test_1k.json")))
    )

    rows = []
    for emb_path in args.emb_paths:
        name = Path(emb_path).stem
        print(f"\n=== {name} ===")
        V, T, ids = load_aligned(emb_path, df)
        stats, c, eps = analyze(name, V, T, args.topk)
        for k, v in stats.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")
        rows.append(stats)
        torch.save(
            {"c_perp": c, "eps": eps, "ids": ids, "stats": stats},
            out_dir / f"{name}_mga_terms.pt",
        )

    pd.DataFrame(rows).to_csv(out_dir / "mga_summary.csv", index=False)
    print(f"\nWrote summary -> {out_dir / 'mga_summary.csv'}")


if __name__ == "__main__":
    main()
