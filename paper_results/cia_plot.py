from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------
# Data
# ---------------------------------------------------------------
dual_encoder = {
    "CLIP (avg.)": {"SSv2": 52.0, "EPIC": 51.0, "Charades": 48.4},
    "DINO.txt": {"SSv2": 52.1, "EPIC": 50.6, "Charades": 50.7},
    "Perception Enc.": {"SSv2": 50.1, "EPIC": 48.5, "Charades": 51.3},
    "InternVideo 2": {"SSv2": 52.5, "EPIC": 48.3, "Charades": 50.7},
}

mllm_finetuned = {
    "VLM2Vec-V2": {"SSv2": 58.8, "EPIC": 49.4, "Charades": 53.5},
    "LAMRA": {"SSv2": 55.3, "EPIC": 53.7, "Charades": 52.1},
    "GVE-7B": {"SSv2": 53.4, "EPIC": 54.7, "Charades": 54.2},
    "E5-V": {"SSv2": 52.6, "EPIC": 57.1, "Charades": 57.1},
    "ArrowRL": {"SSv2": 67.5, "EPIC": 55.7, "Charades": 57.1},
    "CaRe": {"SSv2": 66.4, "EPIC": 62.3, "Charades": 56.1},
}

tara_pairs = [
    (
        "Tarsier2-7B",
        "+ TARA",
        {"SSv2": 77.7, "EPIC": 67.4, "Charades": 60.5},
        {"SSv2": 88.9, "EPIC": 81.1, "Charades": 71.4},
    ),
    (
        "InternVL3-8B",
        "+ TARA",
        {"SSv2": 56.8, "EPIC": 57.5, "Charades": 54.0},
        {"SSv2": 73.1, "EPIC": 68.9, "Charades": 64.1},
    ),
]

DATASETS = ["SSv2", "EPIC", "Charades"]


# ---------------------------------------------------------------
# Poster typography and colors
# ---------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.titlesize": 28,
        "axes.labelsize": 22,
        "xtick.labelsize": 18,
        "ytick.labelsize": 17,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

gray_stops = ["#c8c8c8", "#b0b0b0", "#969696", "#7a7a7a"]
gold_cmap = mcolors.LinearSegmentedColormap.from_list(
    "gold_brown", ["#fff3c4", "#f4bf67", "#b97828"]
)

purple_light = "#eadcf6"
purple_dark = "#673091"
grid_color = "#cfcfcf"
label_color = "#242424"


# ---------------------------------------------------------------
# Layout parameters
# ---------------------------------------------------------------
BAR_H = 0.78
TOUCH_DY = BAR_H
SMALL_GAP_DY = BAR_H * 1.15
GROUP_GAP = BAR_H * 0.9
PAIR_GAP = BAR_H * 0.55

# Row order is fixed from SSv2 (first subplot) so shared y-labels stay aligned.
de_order = sorted(dual_encoder, key=lambda name: dual_encoder[name]["SSv2"])
mllm_order = sorted(mllm_finetuned, key=lambda name: mllm_finetuned[name]["SSv2"])
tara_order = sorted(tara_pairs, key=lambda pair: pair[2]["SSv2"])

# Gray scores are tightly clustered; space colors by rank so bars stay distinct.
de_color = {k: gray_stops[i] for i, k in enumerate(de_order)}

mllm_ssv2_vals = [mllm_finetuned[k]["SSv2"] for k in mllm_order]
mllm_vmin, mllm_vmax = min(mllm_ssv2_vals), max(mllm_ssv2_vals)
mllm_frac = {
    k: ((mllm_finetuned[k]["SSv2"] - mllm_vmin) / (mllm_vmax - mllm_vmin) if mllm_vmax > mllm_vmin else 0.5)
    for k in mllm_order
}


def build_rows(dataset):
    """Visual order, top to bottom. Order is locked to SSv2.

    Dual-encoder and MLLM groups are sorted low -> high (top -> down).
    Each TARA pair is base on top, +TARA directly below.
    """
    rows = []

    for name in de_order:
        rows.append((name, dual_encoder[name][dataset], de_color[name], None))

    for name in mllm_order:
        rows.append((name, mllm_finetuned[name][dataset], gold_cmap(mllm_frac[name]), None))

    for base_name, tara_name, base_vals, tara_vals in tara_order:
        rows.append((base_name, base_vals[dataset], purple_light, "."))
        rows.append((tara_name, tara_vals[dataset], purple_dark, None))

    return rows


def gap_after(visual_i, n_de, n_mllm):
    """Vertical step from this bar to the bar above it."""
    if visual_i >= n_de + n_mllm:
        tara_offset = visual_i - (n_de + n_mllm)
        if tara_offset % 2 == 1:
            return TOUCH_DY
        if tara_offset == 0:
            return GROUP_GAP + BAR_H
        return PAIR_GAP + BAR_H
    if visual_i == n_de:
        return GROUP_GAP + BAR_H
    if visual_i >= n_de:
        return SMALL_GAP_DY
    return TOUCH_DY


def compute_positions(dataset):
    visual = build_rows(dataset)
    n_de = len(dual_encoder)
    n_mllm = len(mllm_finetuned)
    n = len(visual)

    positions_bottom_up = []
    y = 0.0
    for i in range(n):
        positions_bottom_up.append(y)
        visual_i = n - 1 - i
        if i < n - 1:
            y += gap_after(visual_i, n_de, n_mllm)

    return visual, list(reversed(positions_bottom_up))


def main():
    row_cache = {}
    all_max = 0
    for dataset in DATASETS:
        rows, positions = compute_positions(dataset)
        row_cache[dataset] = (rows, positions)
        all_max = max(all_max, max(v for _, v, _, _ in rows))

    fig, axes = plt.subplots(1, 3, figsize=(19, 10.5), sharex=True, sharey=False)
    fig.patch.set_facecolor("none")

    x_min = 25
    x_max = np.ceil(all_max / 5) * 5 + 7
    first_positions = row_cache[DATASETS[0]][1]
    first_labels = [r[0] for r in row_cache[DATASETS[0]][0]]
    ymin = min(first_positions) - BAR_H
    ymax = max(first_positions) + BAR_H

    for i, (ax, dataset) in enumerate(zip(axes, DATASETS)):
        rows, positions = row_cache[dataset]
        values = [r[1] for r in rows]
        colors = [r[2] for r in rows]
        hatches = [r[3] for r in rows]

        for y, value, color, hatch in zip(positions, values, colors, hatches):
            ax.barh(
                y,
                value,
                height=BAR_H,
                color=color,
                edgecolor=purple_dark if hatch else "#4a4a4a",
                linewidth=0.7,
                hatch=hatch,
                zorder=3,
            )

        for y, value in zip(positions, values):
            ax.text(
                value + 0.9,
                y,
                f"{value:.1f}",
                va="center",
                ha="left",
                fontsize=19,
                fontweight="bold",
                color=label_color,
            )

        ax.axvline(50, color="#d62728", linestyle="--", linewidth=3, zorder=4)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel(dataset, labelpad=12, fontweight="bold")
        ax.set_yticks(first_positions)
        if i == 0:
            ax.set_yticklabels(first_labels, color=label_color, fontsize=20, fontweight="bold")
            ax.tick_params(axis="y", length=0, pad=7, labelleft=True)
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0, labelleft=False)
        ax.tick_params(axis="x", length=4, width=1.0, color="#444444")
        ax.xaxis.grid(True, linestyle=":", linewidth=1.1, color=grid_color, zorder=0)
        ax.set_axisbelow(True)

        ax.set_facecolor("none")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(True)
        ax.spines["left"].set_linewidth(2)
        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["bottom"].set_color("#555555")

    plt.tight_layout()
    plt.subplots_adjust(left=0.20, right=0.985, top=0.98, bottom=0.10, wspace=0.08)

    out_dir = Path("paper_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "chiral_grouped_subplots_eccv.png"
    pdf_path = out_dir / "chiral_grouped_subplots_eccv.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight", facecolor="none", transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
