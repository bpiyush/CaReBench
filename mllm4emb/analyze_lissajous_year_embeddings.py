#!/usr/bin/env python3
"""Fit Lissajous curves to PCA projections of year embeddings.

Propositions 1 and 3 predict that projections onto any two principal components
follow Lissajous curves:

    (x(t), y(t)) = (A sin(a t), B sin(b t + δ))

where t indexes position along the temporal / sequential axis (here: year order).
"""
import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from sklearn.decomposition import PCA

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


@dataclass
class LissajousFit:
    embed_path: str
    label: str
    pc_x: int
    pc_y: int
    A: float
    a: float
    B: float
    b: float
    delta: float
    rmse: float
    r2: float
    r2_x: float
    r2_y: float
    nrmse: float
    rss: float
    n_points: int
    n_params: int
    aic: float
    bic: float
    linear_r2: float
    linear_rmse: float
    aic_improvement_vs_linear: float


def load_embeddings(path: str) -> Tuple[np.ndarray, np.ndarray, str]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    years = np.asarray(data["years"], dtype=float)
    Z = data["embeddings"]
    if torch.is_tensor(Z):
        Z = Z.numpy()
    label = data.get("model") or data.get("model_id") or os.path.splitext(os.path.basename(path))[0]
    return years, np.asarray(Z, dtype=float), str(label)


def time_parameter(years: np.ndarray) -> np.ndarray:
    order = np.argsort(years)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(years))
    if len(years) == 1:
        return ranks
    return ranks / (len(years) - 1)


def lissajous_xy(t: np.ndarray, A: float, a: float, B: float, b: float, delta: float) -> Tuple[np.ndarray, np.ndarray]:
    x = A * np.sin(a * t)
    y = B * np.sin(b * t + delta)
    return x, y


def _dominant_angular_freq(signal: np.ndarray, t: np.ndarray) -> float:
    y = signal - signal.mean()
    if len(y) < 4:
        return 2 * np.pi
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    spectrum = np.abs(np.fft.rfft(y))[1:]
    if spectrum.size == 0:
        return 2 * np.pi
    k = int(np.argmax(spectrum)) + 1
    freq_cycles = k / (len(y) * dt)
    return 2 * np.pi * freq_cycles


def _initial_guesses(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> List[np.ndarray]:
    guesses = []
    a0 = _dominant_angular_freq(x, t)
    b0 = _dominant_angular_freq(y, t)
    A0 = np.std(x) * np.sqrt(2.0)
    B0 = np.std(y) * np.sqrt(2.0)
    base = np.array([max(A0, 1e-6), a0, max(B0, 1e-6), b0, 0.0], dtype=float)
    guesses.append(base)

    for k in range(1, 8):
        for l in range(1, 8):
            guesses.append(
                np.array([max(A0, 1e-6), 2 * np.pi * k, max(B0, 1e-6), 2 * np.pi * l, 0.0], dtype=float)
            )
    return guesses


def _lissajous_residuals(params: np.ndarray, t: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    A, a, B, b, delta = params
    x_hat, y_hat = lissajous_xy(t, A, a, B, b, delta)
    return np.concatenate([x - x_hat, y - y_hat])


def fit_lissajous(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, float]:
    best = None
    best_cost = np.inf
    bounds = ([0.0, 0.0, 0.0, 0.0, -np.pi], [np.inf, 50 * np.pi, np.inf, 50 * np.pi, np.pi])

    for p0 in _initial_guesses(x, y, t):
        try:
            result = least_squares(
                _lissajous_residuals,
                p0,
                args=(t, x, y),
                bounds=bounds,
                max_nfev=5000,
            )
        except Exception:
            continue
        if result.cost < best_cost:
            best_cost = result.cost
            best = result
    if best is None:
        raise RuntimeError("Lissajous optimization failed for all initializations.")
    return best.x, float(best.cost)


def fit_linear(x: np.ndarray, y: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    tx = np.column_stack([np.ones_like(t), t])
    coef_x = np.linalg.lstsq(tx, x, rcond=None)[0]
    coef_y = np.linalg.lstsq(tx, y, rcond=None)[0]
    x_hat = tx @ coef_x
    y_hat = tx @ coef_y
    return x_hat, y_hat


def goodness_of_fit(
    x: np.ndarray,
    y: np.ndarray,
    x_hat: np.ndarray,
    y_hat: np.ndarray,
    n_params: int,
) -> Dict[str, float]:
    resid = np.column_stack([x - x_hat, y - y_hat])
    rss = float(np.sum(resid ** 2))
    n = len(x)
    tot = float(np.sum((x - x.mean()) ** 2 + (y - y.mean()) ** 2))
    r2 = 1.0 - rss / tot if tot > 0 else float("nan")
    rss_x = float(np.sum((x - x_hat) ** 2))
    rss_y = float(np.sum((y - y_hat) ** 2))
    tot_x = float(np.sum((x - x.mean()) ** 2))
    tot_y = float(np.sum((y - y.mean()) ** 2))
    r2_x = 1.0 - rss_x / tot_x if tot_x > 0 else float("nan")
    r2_y = 1.0 - rss_y / tot_y if tot_y > 0 else float("nan")
    rmse = float(np.sqrt(rss / n))
    scale = float(np.sqrt(np.mean(x ** 2 + y ** 2)))
    nrmse = rmse / scale if scale > 0 else float("nan")
    # Gaussian noise model with 2 coordinates per observation.
    sigma2 = rss / (2 * n)
    loglik = -n * (np.log(2 * np.pi * sigma2) + 1.0) if sigma2 > 0 else float("nan")
    aic = 2 * n_params - 2 * loglik
    bic = n_params * np.log(2 * n) - 2 * loglik
    return {
        "rss": rss,
        "rmse": rmse,
        "r2": r2,
        "r2_x": r2_x,
        "r2_y": r2_y,
        "nrmse": nrmse,
        "aic": aic,
        "bic": bic,
    }


def analyze_pair(
    embed_path: str,
    label: str,
    coords: np.ndarray,
    years: np.ndarray,
    t: np.ndarray,
    pc_x: int,
    pc_y: int,
) -> LissajousFit:
    x = coords[:, pc_x]
    y = coords[:, pc_y]
    params, _ = fit_lissajous(x, y, t)
    A, a, B, b, delta = params
    x_hat, y_hat = lissajous_xy(t, A, a, B, b, delta)
    metrics = goodness_of_fit(x, y, x_hat, y_hat, n_params=5)

    lin_x, lin_y = fit_linear(x, y, t)
    linear_metrics = goodness_of_fit(x, y, lin_x, lin_y, n_params=4)

    return LissajousFit(
        embed_path=embed_path,
        label=label,
        pc_x=pc_x + 1,
        pc_y=pc_y + 1,
        A=float(A),
        a=float(a),
        B=float(B),
        b=float(b),
        delta=float(delta),
        rmse=metrics["rmse"],
        r2=metrics["r2"],
        r2_x=metrics["r2_x"],
        r2_y=metrics["r2_y"],
        nrmse=metrics["nrmse"],
        rss=metrics["rss"],
        n_points=len(x),
        n_params=5,
        aic=metrics["aic"],
        bic=metrics["bic"],
        linear_r2=linear_metrics["r2"],
        linear_rmse=linear_metrics["rmse"],
        aic_improvement_vs_linear=float(linear_metrics["aic"] - metrics["aic"]),
    )


def plot_lissajous_fit(
    coords: np.ndarray,
    years: np.ndarray,
    t: np.ndarray,
    fit: LissajousFit,
    out_path: str,
):
    pc_x, pc_y = fit.pc_x - 1, fit.pc_y - 1
    x = coords[:, pc_x]
    y = coords[:, pc_y]

    t_dense = np.linspace(t.min(), t.max(), 2000)
    x_curve, y_curve = lissajous_xy(t_dense, fit.A, fit.a, fit.B, fit.b, fit.delta)

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="white")
    order = np.argsort(years)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(years)))

    ax.scatter(x, y, c=colors, s=18, edgecolors="none", alpha=0.85, zorder=2)
    ax.plot(x_curve, y_curve, color="0.2", lw=1.8, alpha=0.9, zorder=3, label="Lissajous fit")
    ax.set_xlabel(f"PCA axis {fit.pc_x}")
    ax.set_ylabel(f"PCA axis {fit.pc_y}")
    ax.set_title(
        f"{fit.label}: PC{fit.pc_x} vs PC{fit.pc_y}\n"
        f"$R^2$={fit.r2:.3f}, RMSE={fit.rmse:.3f}, "
        f"linear $R^2$={fit.linear_r2:.3f}"
    )
    ax.grid(alpha=0.25)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def analyze_embeddings(
    embed_path: str,
    n_components: int,
    pc_pairs: Sequence[Tuple[int, int]],
    out_dir: str,
) -> List[LissajousFit]:
    years, Z, label = load_embeddings(embed_path)
    t = time_parameter(years)
    coords = PCA(n_components=n_components).fit_transform(Z)

    fits: List[LissajousFit] = []
    stem = os.path.splitext(os.path.basename(embed_path))[0]
    for pc_x, pc_y in pc_pairs:
        fit = analyze_pair(embed_path, label, coords, years, t, pc_x, pc_y)
        fits.append(fit)
        plot_path = os.path.join(
            out_dir,
            f"{stem}_lissajous_pc{fit.pc_x}_pc{fit.pc_y}.pdf",
        )
        plot_lissajous_fit(coords, years, t, fit, plot_path)
        print(
            f"[{label}] PC{fit.pc_x}-PC{fit.pc_y}: "
            f"R²={fit.r2:.4f}, RMSE={fit.rmse:.4f}, "
            f"linear R²={fit.linear_r2:.4f}, ΔAIC={fit.aic_improvement_vs_linear:.1f}"
        )
    return fits


def parse_pc_pairs(spec: str, n_components: int) -> List[Tuple[int, int]]:
    if spec == "all":
        return list(combinations(range(n_components), 2))
    pairs = []
    for item in spec.split(","):
        i, j = map(int, item.split("-"))
        pairs.append((i - 1, j - 1))
    return pairs


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit Lissajous curves to PCA projections of year embeddings."
    )
    parser.add_argument(
        "--embed_paths",
        nargs="+",
        default=[
            "year_embeddings.pt",
            "year_embeddings_qwen3vl.pt",
            "year_embeddings_clip.pt",
            "year_embeddings_dinotxt.pt",
            "year_embeddings_siglip2.pt",
        ],
        help="Paths to .pt files saved by embed_year*.py scripts.",
    )
    parser.add_argument("--n_components", type=int, default=3)
    parser.add_argument(
        "--pc_pairs",
        type=str,
        default="all",
        help='PC index pairs to analyze, e.g. "1-2,1-3,2-3" or "all".',
    )
    parser.add_argument("--out_dir", type=str, default="lissajous_year_analysis")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pc_pairs = parse_pc_pairs(args.pc_pairs, args.n_components)

    all_fits: List[LissajousFit] = []
    for path in args.embed_paths:
        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue
        all_fits.extend(
            analyze_embeddings(path, args.n_components, pc_pairs, args.out_dir)
        )

    if not all_fits:
        raise FileNotFoundError("No embedding files were analyzed.")

    df = pd.DataFrame([asdict(f) for f in all_fits])
    csv_path = os.path.join(args.out_dir, "lissajous_fit_metrics.csv")
    json_path = os.path.join(args.out_dir, "lissajous_fit_metrics.json")
    df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump([asdict(f) for f in all_fits], f, indent=2)

    print(f"\nWrote metrics: {csv_path}")
    print(f"Wrote metrics: {json_path}")
    print("\nSummary (sorted by R²):")
    summary = df.sort_values("r2", ascending=False)[
        ["label", "pc_x", "pc_y", "r2", "rmse", "nrmse", "linear_r2", "aic_improvement_vs_linear"]
    ]
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
