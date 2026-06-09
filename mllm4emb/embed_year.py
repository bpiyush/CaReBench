import os

os.environ["TOKENIZERS_PARALLELISM"] = "False"

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

import shared.utils as su
from models.modeling_encoders import AutoEncoder


YEAR_START = 1700
YEAR_END = 2020


def embed_year_sentence(encoder, year: int) -> torch.Tensor:
    prompt = f"In the year {year}"
    with torch.no_grad():
        zt = encoder.encode_text(prompt).cpu().squeeze(0).float()
        zt = torch.nn.functional.normalize(zt, dim=-1)
    return zt


def compute_year_embeddings(encoder, year_start=YEAR_START, year_end=YEAR_END):
    years = list(range(year_start, year_end + 1))
    embeds = []
    for year in su.log.tqdm_iterator(years, desc="Computing TARA embeddings"):
        embeds.append(embed_year_sentence(encoder, year))
    Z = torch.stack(embeds)
    return years, Z


def visualize_year_embeddings_3d_pca(
    Z,
    years,
    out_path=None,
    show=True,
    elev=22,
    azim=-58,
    figsize=(7, 6),
):
    from sklearn.decomposition import PCA

    X = Z.numpy() if torch.is_tensor(Z) else np.asarray(Z)
    coords = PCA(n_components=3).fit_transform(X)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(years)))

    plt.rcParams.update({"font.family": "serif"})
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_subplot(111, projection="3d", facecolor="white")

    z_floor = coords[:, 2].min()
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        np.full(len(coords), z_floor),
        c="0.75",
        s=14,
        alpha=0.35,
        depthshade=False,
    )
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=colors,
        s=28,
        depthshade=True,
        edgecolors="none",
    )

    ax.set_xlabel("PCA axis 1")
    ax.set_ylabel("PCA axis 2")
    ax.set_zlabel("PCA axis 3")
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("0.85")
    ax.yaxis.pane.set_edgecolor("0.85")
    ax.zaxis.pane.set_edgecolor("0.85")

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote visualization: {out_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and visualize TARA embeddings for 'In the year x' sentences."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/",
    )
    parser.add_argument("--year_start", type=int, default=YEAR_START)
    parser.add_argument("--year_end", type=int, default=YEAR_END)
    parser.add_argument(
        "--out_plot",
        type=str,
        default="year_embeddings_3d_pca.pdf",
        help="Output path for the 3D PCA visualization.",
    )
    parser.add_argument("--elev", type=float, default=22, help="3D view elevation.")
    parser.add_argument("--azim", type=float, default=-58, help="3D view azimuth.")
    parser.add_argument(
        "--out_embeds",
        type=str,
        default="year_embeddings.pt",
        help="Output path for saved embeddings.",
    )
    parser.add_argument(
        "--load_embeds",
        type=str,
        default=None,
        help="If set, skip encoding and load embeddings from this .pt file.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display the plot interactively.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.load_embeds is not None:
        data = torch.load(args.load_embeds, map_location="cpu")
        years = data["years"]
        Z = data["embeddings"]
        print(f"Loaded embeddings from {args.load_embeds}: {Z.shape}")
    else:
        encoder = AutoEncoder.from_pretrained(
            args.model_path,
            device_map="auto",
            attn_implementation="flash_attention_2",
            dtype=torch.bfloat16,
        )
        su.misc.num_params(encoder.model)

        years, Z = compute_year_embeddings(
            encoder, year_start=args.year_start, year_end=args.year_end
        )
        print(f"Computed embeddings: {Z.shape}")

        torch.save({"years": years, "embeddings": Z}, args.out_embeds)
        print(f"Saved embeddings: {args.out_embeds}")

    visualize_year_embeddings_3d_pca(
        Z,
        years,
        out_path=args.out_plot,
        show=not args.no_show,
        elev=args.elev,
        azim=args.azim,
    )


if __name__ == "__main__":
    main()
