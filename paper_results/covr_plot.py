from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import blended_transform_factory


# ---------------------------------------------------------------
# Data: WebVid-CoVR R@1 (Table 5)
# ---------------------------------------------------------------
# Zero-shot vs fine-tuned variants stay in the same model-family group.
# Two fine-tuned BLIP (V+T) rows appear in the table (50.6 and 51.8).
GROUPS = [
    {
        "name": "BLIP",
        "bars": [
            ("(T) ZS", 19.7),
            ("(T) FT", 23.7),
            ("(V) ZS", 34.9),
            ("(V) FT", 38.9),
            ("(V+T) ZS", 45.5),
            ("(V+T) FT", 50.6),
            ("(V+T) FT*", 51.8),
        ],
    },
    {
        "name": "CLIP",
        "bars": [
            ("(V+T) ZS", 44.4),
            ("(V+T) FT", 50.6),
        ],
    },
    {
        "name": "Ventura et al.",
        "bars": [
            ("[51]", 53.1),
            ("[50]", 59.8),
        ],
    },
    {
        "name": "Tarsier 2",
        "bars": [
            ("+ TARA", 66.3),
        ],
    },
]


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
        "xtick.labelsize": 16,
        "ytick.labelsize": 17,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

gold_cmap = mcolors.LinearSegmentedColormap.from_list(
    "gold_brown", ["#fff3c4", "#f4bf67", "#b97828"]
)
gray_cmap = mcolors.LinearSegmentedColormap.from_list(
    "gray_ramp", ["#c8c8c8", "#7a7a7a"]
)
orange_cmap = mcolors.LinearSegmentedColormap.from_list(
    "ventura_orange", ["#f8d4a8", "#e07a3d", "#b85a1a"]
)
purple_cmap = mcolors.LinearSegmentedColormap.from_list(
    "tara_purple", ["#eadcf6", "#673091"]
)

GROUP_CMAPS = {
    "BLIP": gold_cmap,
    "CLIP": gray_cmap,
    "Ventura et al.": orange_cmap,
    "Tarsier 2": purple_cmap,
}

grid_color = "#cfcfcf"
label_color = "#242424"
edge_color = "#4a4a4a"
tara_edge = "#673091"


# ---------------------------------------------------------------
# Layout parameters
# ---------------------------------------------------------------
BAR_W = 0.78
WITHIN_DX = BAR_W * 1.15
GROUP_GAP = BAR_W * 1.35


def color_for(cmap, value, vmin, vmax):
    if vmax <= vmin:
        return cmap(0.85)
    frac = (value - vmin) / (vmax - vmin)
    # Keep the lightest bar readable; push the range toward mid/dark.
    return cmap(0.18 + 0.82 * frac)


def compute_layout():
    rows = []
    x = 0.0
    group_spans = []

    for group in GROUPS:
        bars = sorted(group["bars"], key=lambda item: item[1])
        scores = [v for _, v in bars]
        vmin, vmax = min(scores), max(scores)
        cmap = GROUP_CMAPS[group["name"]]
        start = x
        for label, value in bars:
            color = color_for(cmap, value, vmin, vmax)
            rows.append((group["name"], label, value, color, x))
            x += WITHIN_DX
        x -= WITHIN_DX
        group_spans.append((group["name"], start, x))
        x += GROUP_GAP + BAR_W

    return rows, group_spans


def main():
    rows, group_spans = compute_layout()
    xs = [r[4] for r in rows]
    values = [r[2] for r in rows]
    labels = [r[1] for r in rows]
    colors = [r[3] for r in rows]
    names = [r[0] for r in rows]

    fig, ax = plt.subplots(figsize=(16.5, 8.8))
    fig.patch.set_facecolor("none")

    for x, value, color, name in zip(xs, values, colors, names):
        ax.bar(
            x,
            value,
            width=BAR_W,
            color=color,
            edgecolor=tara_edge if name == "Tarsier 2" else edge_color,
            linewidth=0.7,
            zorder=3,
        )

    y_max = np.ceil(max(values) / 5) * 5 + 7
    for x, value in zip(xs, values):
        ax.text(
            x,
            value + 0.9,
            f"{value:.1f}",
            va="bottom",
            ha="center",
            fontsize=16,
            fontweight="bold",
            color=label_color,
        )

    ax.set_xlim(min(xs) - BAR_W, max(xs) + BAR_W)
    ax.set_ylim(0, y_max)
    ax.set_ylabel("R@1", labelpad=10, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, color=label_color, fontsize=15, fontweight="bold")
    ax.tick_params(axis="x", length=0, pad=6)
    ax.tick_params(axis="y", length=4, width=1.0, color="#444444")
    ax.yaxis.grid(True, linestyle=":", linewidth=1.1, color=grid_color, zorder=0)
    ax.set_axisbelow(True)

    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["bottom"].set_color("#555555")

    trans = blended_transform_factory(ax.transData, ax.transAxes)
    for name, x0, x1 in group_spans:
        ax.plot(
            [x0 - 0.22, x1 + 0.22],
            [-0.16, -0.16],
            color="#555555",
            linewidth=1.6,
            clip_on=False,
            transform=trans,
        )
        ax.text(
            0.5 * (x0 + x1),
            -0.21,
            name,
            ha="center",
            va="top",
            fontsize=18,
            fontweight="bold",
            color=label_color,
            transform=trans,
            clip_on=False,
        )

    plt.tight_layout()
    plt.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.24)

    out_dir = Path("paper_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "covr_grouped_r1_eccv.png"
    pdf_path = out_dir / "covr_grouped_r1_eccv.pdf"
    fig.savefig(png_path, dpi=350, bbox_inches="tight", facecolor="none", transparent=True)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved {png_path}")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
