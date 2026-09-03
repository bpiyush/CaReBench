"""
Estimate Modality Gap Assumption (MGA) terms on SSv2 validation pairs.

For each video–text pair (v, t):
    v - t = c_perp + ε

Uses base Tarsier2-7B video embeddings from the nuanced-retrieval cache and
encodes SSv2 *detailed* captions (labels/all.csv `label` field), not templates.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = ROOT / "data/nuanced_retrieval_data-validation-v1.csv"
DEFAULT_SSV2_LABELS = Path("/scratch/shared/beegfs/piyush/datasets/SSv2/labels/all.csv")
DEFAULT_VIDEO_EMB = Path(
    "/work/piyush/experiments/CaRe/Tarsier2-7b-0115/special_milestones/"
    "Tarsier2-TARA-chiral10k_covr10k/embs/"
    "tarsier2_7b_nuanced_retrieval_data-validation-v1_embeddings.pt"
)
DEFAULT_MODEL = "/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115"
DEFAULT_OUT = ROOT / "outputs/mga_ssv2_base"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv_path", type=str, default=str(DEFAULT_CSV))
    p.add_argument("--ssv2_labels", type=str, default=str(DEFAULT_SSV2_LABELS))
    p.add_argument("--video_emb_path", type=str, default=str(DEFAULT_VIDEO_EMB))
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL)
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT))
    p.add_argument(
        "--text_emb_cache",
        type=str,
        default=None,
        help="Optional .pt cache for detailed-caption embeddings "
        "(default: {out_dir}/ssv2_detailed_caption_embeddings_base.pt)",
    )
    p.add_argument("--n_pcs", type=int, default=50, help="PCs used for orthogonality checks")
    p.add_argument("--skip_encode", action="store_true", help="Fail if text cache missing")
    return p.parse_args()


def build_pairs(csv_path: str, labels_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    vids = (
        df[(df["source"] == "cia-ssv2") & (df["modality"] == "video")]["id"]
        .astype(str)
        .unique()
        .tolist()
    )
    lab = pd.read_csv(labels_path, dtype={"id": str})
    lab = lab.drop_duplicates("id").set_index("id")
    missing = [v for v in vids if v not in lab.index]
    if missing:
        raise RuntimeError(f"{len(missing)} video ids missing from SSv2 labels, e.g. {missing[:5]}")

    rows = []
    for vid in vids:
        rows.append(
            {
                "video_id": vid,
                "caption": str(lab.loc[vid, "label"]),
                "template": str(lab.loc[vid, "template"]),
            }
        )
    return pd.DataFrame(rows)


def encode_captions(captions: list[str], model_path: str) -> dict[str, torch.Tensor]:
    import sys

    sys.path.insert(0, str(ROOT))
    from models.modeling_encoders import AutoEncoder

    model = AutoEncoder.from_pretrained(
        model_path,
        device_map="auto",
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    out: dict[str, torch.Tensor] = {}
    unique = list(dict.fromkeys(captions))
    print(f"Encoding {len(unique)} unique detailed captions with {model_path}")
    for i, cap in enumerate(unique):
        z = model.encode_text(cap).squeeze(0).detach().float().cpu()
        z = F.normalize(z, dim=-1)
        out[cap] = z
        if (i + 1) % 50 == 0 or i + 1 == len(unique):
            print(f"  encoded {i + 1}/{len(unique)}")
    return out


def stack_normalized(embs: dict, keys: list[str]) -> torch.Tensor:
    xs = []
    for k in keys:
        z = embs[k].float()
        xs.append(F.normalize(z, dim=-1))
    return torch.stack(xs, dim=0)


def pca_basis(X: torch.Tensor, n_pcs: int) -> torch.Tensor:
    """Return top-n_pcs right singular vectors of centered X (D x k)."""
    Xc = X - X.mean(dim=0, keepdim=True)
    # economy SVD on (N x D); Vh[:k] are top PCs in R^D
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    k = min(n_pcs, Vh.shape[0])
    return Vh[:k].T.contiguous()  # (D, k)


def subspace_energy(vec: torch.Tensor, basis: torch.Tensor) -> dict:
    """Fraction of ||vec||^2 explained by orthonormal-ish PCA basis columns."""
    # re-orthonormalize numerically
    Q, _ = torch.linalg.qr(basis, mode="reduced")
    proj = Q @ (Q.T @ vec)
    residual = vec - proj
    n2 = vec.norm().item() ** 2 + 1e-12
    return {
        "cos_to_subspace": (proj.norm() / (vec.norm() + 1e-12)).item(),
        "frac_energy_in_subspace": (proj.norm().item() ** 2 / n2),
        "frac_energy_orthogonal": (residual.norm().item() ** 2 / n2),
        "residual_norm": residual.norm().item(),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    text_cache = Path(
        args.text_emb_cache
        or (out_dir / "ssv2_detailed_caption_embeddings_base.pt")
    )

    pairs = build_pairs(args.csv_path, args.ssv2_labels)
    print(f"SSv2 pairs: {len(pairs)} | unique captions: {pairs['caption'].nunique()}")

    video_emb_all = torch.load(args.video_emb_path, map_location="cpu", weights_only=False)
    video_keys = pairs["video_id"].tolist()
    missing_v = [k for k in video_keys if k not in video_emb_all]
    if missing_v:
        raise RuntimeError(f"Missing video embeddings: {missing_v[:5]} ({len(missing_v)} total)")

    if text_cache.exists():
        print(f"Loading text embedding cache: {text_cache}")
        text_emb = torch.load(text_cache, map_location="cpu", weights_only=False)
    else:
        if args.skip_encode:
            raise FileNotFoundError(text_cache)
        text_emb = encode_captions(pairs["caption"].tolist(), args.model_path)
        torch.save(text_emb, text_cache)
        print(f"Saved text embeddings -> {text_cache}")

    missing_t = [c for c in pairs["caption"] if c not in text_emb]
    if missing_t:
        # encode only missing
        if args.skip_encode:
            raise RuntimeError(f"{len(missing_t)} captions missing from cache")
        extra = encode_captions(missing_t, args.model_path)
        text_emb.update(extra)
        torch.save(text_emb, text_cache)

    V = stack_normalized(video_emb_all, video_keys)
    T = stack_normalized(text_emb, pairs["caption"].tolist())
    assert V.shape == T.shape
    N, D = V.shape
    print(f"Stacked embeddings: N={N}, D={D}")
    print(f"Video L2 norms: min={V.norm(dim=-1).min():.4f} max={V.norm(dim=-1).max():.4f}")
    print(f"Text  L2 norms: min={T.norm(dim=-1).min():.4f} max={T.norm(dim=-1).max():.4f}")

    # MGA: v - t = c_perp + ε  =>  MLE under isotropic Gaussian noise
    delta = V - T
    c_perp = delta.mean(dim=0)
    eps = delta - c_perp.unsqueeze(0)

    # noise stats
    eps_var_per_dim = eps.var(dim=0, unbiased=True)
    nu2 = eps_var_per_dim.mean().item()
    summary = {
        "n_pairs": N,
        "dim": D,
        "unique_captions": int(pairs["caption"].nunique()),
        "c_perp_l2": c_perp.norm().item(),
        "mean_delta_l2": delta.norm(dim=-1).mean().item(),
        "std_delta_l2": delta.norm(dim=-1).std().item(),
        "mean_eps_l2": eps.norm(dim=-1).mean().item(),
        "std_eps_l2": eps.norm(dim=-1).std().item(),
        "nu2_mean_var_per_dim": nu2,
        "nu": float(np.sqrt(nu2)),
        "frac_delta_explained_by_c": float(
            (c_perp.norm().item() ** 2)
            / (delta.pow(2).mean(dim=0).sum().item() + 1e-12)
        ),
        "mean_abs_cos_c_with_v": float(
            (V @ c_perp).abs().mean().item() / (c_perp.norm().item() + 1e-12)
        ),
        "mean_abs_cos_c_with_t": float(
            (T @ c_perp).abs().mean().item() / (c_perp.norm().item() + 1e-12)
        ),
        "cos_c_with_mean_v": float(
            F.cosine_similarity(c_perp.unsqueeze(0), V.mean(0).unsqueeze(0)).item()
        ),
        "cos_c_with_mean_t": float(
            F.cosine_similarity(c_perp.unsqueeze(0), T.mean(0).unsqueeze(0)).item()
        ),
        "mean_v_minus_mean_t_l2": (V.mean(0) - T.mean(0)).norm().item(),
        "video_emb_path": args.video_emb_path,
        "model_path": args.model_path,
        "ssv2_labels": args.ssv2_labels,
        "text_emb_cache": str(text_cache),
    }

    # Orthogonality vs video / text / joint principal subspaces
    for name, X in [("video", V), ("text", T), ("joint", torch.cat([V, T], dim=0))]:
        basis = pca_basis(X, args.n_pcs)
        stats = subspace_energy(c_perp, basis)
        summary[f"c_vs_{name}_top{args.n_pcs}_pcs"] = stats

    # Also report residual after removing components in joint top-PCs
    joint_basis = pca_basis(torch.cat([V, T], dim=0), args.n_pcs)
    Q, _ = torch.linalg.qr(joint_basis, mode="reduced")
    c_orth = c_perp - Q @ (Q.T @ c_perp)
    summary["c_perp_after_removing_joint_pcs_l2"] = c_orth.norm().item()
    summary["frac_c_outside_joint_pcs"] = (
        c_orth.norm().item() ** 2 / (c_perp.norm().item() ** 2 + 1e-12)
    )

    # Save tensors
    torch.save(
        {
            "video_ids": video_keys,
            "captions": pairs["caption"].tolist(),
            "templates": pairs["template"].tolist(),
            "V": V,
            "T": T,
            "c_perp": c_perp,
            "eps": eps,
            "delta": delta,
            "c_perp_joint_pc_orthogonalized": c_orth,
        },
        out_dir / "mga_ssv2_base_tensors.pt",
    )
    with open(out_dir / "mga_ssv2_base_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-pair error table
    pair_df = pairs.copy()
    pair_df["delta_l2"] = delta.norm(dim=-1).numpy()
    pair_df["eps_l2"] = eps.norm(dim=-1).numpy()
    pair_df["cos_v_t"] = F.cosine_similarity(V, T).numpy()
    pair_df.to_csv(out_dir / "mga_ssv2_base_per_pair.csv", index=False)

    print("\n=== MGA summary (base Tarsier2, SSv2 detailed captions) ===")
    for k in [
        "n_pairs",
        "unique_captions",
        "c_perp_l2",
        "mean_delta_l2",
        "mean_eps_l2",
        "nu",
        "mean_abs_cos_c_with_v",
        "mean_abs_cos_c_with_t",
        "frac_c_outside_joint_pcs",
        "c_perp_after_removing_joint_pcs_l2",
    ]:
        print(f"  {k}: {summary[k]}")
    print(f"\nWrote:\n  {out_dir / 'mga_ssv2_base_summary.json'}\n  {out_dir / 'mga_ssv2_base_tensors.pt'}\n  {out_dir / 'mga_ssv2_base_per_pair.csv'}")


if __name__ == "__main__":
    main()
