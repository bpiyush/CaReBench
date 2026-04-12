#!/usr/bin/env python3
"""Plot validation metric vs ensemble alpha from ensemble_alpha_sweep_summary.txt."""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt


def load_summary(path: str) -> tuple[list[float], list[float]]:
    alphas: list[float] = []
    metrics: list[float] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                a = float(parts[0])
                m = float(parts[1])
            except ValueError:
                continue
            if m != m:  # skip NaN
                continue
            alphas.append(a)
            metrics.append(m)
    return alphas, metrics


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_summary = os.path.join(
        root, "_ensemble_alpha_sweep_validation", "ensemble_alpha_sweep_summary.txt"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        type=str,
        default=default_summary,
        help="Path to ensemble_alpha_sweep_summary.txt",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output image path (default: same dir as summary, .png)",
    )
    args = parser.parse_args()

    alphas, metrics = load_summary(args.summary)
    if not alphas:
        raise SystemExit(f"No data rows parsed from {args.summary}")

    out_path = args.out
    if not out_path:
        base, _ = os.path.splitext(args.summary)
        out_path = base + "_plot.png"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Times New Roman",
                "Nimbus Roman",
                "Times",
                "DejaVu Serif",
                "Bitstream Vera Serif",
                "Computer Modern Roman",
            ],
            "mathtext.fontset": "dejavuserif",
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 150,
        }
    )

    fig, ax = plt.subplots(figsize=(6.0, 3.8), layout="constrained")
    ax.plot(alphas, metrics, color="#1a5276", linewidth=2.0, marker="o", markersize=4.5)
    ax.set_xlabel(r"$\alpha$ (weight on TARA)")
    ax.set_ylabel("Avg. validation metric (%)")
    ax.set_title(
        "TARA + Qwen3-VL-E ensemble\n"
        "mean of SSv2 v2t R@1 (3), MSRVTT neg R@5 (2), CoVR R@1/R@5"
    )
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.85)
    ax.set_axisbelow(True)

    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
