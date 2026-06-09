#!/users/piyush/miniconda3/envs/qwen/bin/python
"""Embed 'In the year x' with CLIP, DINO.txt, and/or SigLIP2.

Run via ``mllm4emb/run_embed_year_vlm.sh`` or the shebang above (``qwen`` conda env).
"""
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "False"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

import shared.utils as su


YEAR_START = 1700
YEAR_END = 2020

MODEL_CHOICES = ("clip", "dinotxt", "siglip2")
MODEL_DEFAULTS = {
    "clip": "ViT-L/14",
    "dinotxt": "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
    "siglip2": "google/siglip2-so400m-patch14-384",
}


@dataclass
class TextEncoder:
    name: str
    model_id: str
    encode_batch: Callable[[Sequence[str]], torch.Tensor]
    cleanup: Callable[[], None] | None = None


def year_sentences(year_start: int, year_end: int) -> List[str]:
    return [f"In the year {year}" for year in range(year_start, year_end + 1)]


def load_clip_encoder(model_id: str, device: torch.device) -> TextEncoder:
    import clip

    model, _ = clip.load(model_id, device=device)
    model.eval()

    def encode_batch(texts: Sequence[str]) -> torch.Tensor:
        tokens = clip.tokenize(list(texts), truncate=True).to(device)
        with torch.no_grad():
            z = model.encode_text(tokens).float()
            z = F.normalize(z, dim=-1)
        return z.cpu()

    return TextEncoder("clip", model_id, encode_batch)


def load_dinotxt_encoder(model_id: str, device: torch.device) -> TextEncoder:
    model = torch.hub.load("facebookresearch/dinov2", model_id, trust_repo=True)
    model = model.to(device)
    model.eval()
    su.misc.num_params(model)

    from dinov2.hub.dinotxt import get_tokenizer

    tokenizer = get_tokenizer()

    def encode_batch(texts: Sequence[str]) -> torch.Tensor:
        tokens = tokenizer.tokenize(list(texts)).to(device)
        with torch.autocast(device.type, dtype=torch.float):
            with torch.no_grad():
                z = model.encode_text(tokens).float()
                z = F.normalize(z, dim=-1)
        return z.cpu()

    return TextEncoder("dinotxt", model_id, encode_batch)


def load_siglip2_encoder(model_id: str, device: torch.device) -> TextEncoder:
    from transformers import AutoModel, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    model.eval()

    def encode_batch(texts: Sequence[str]) -> torch.Tensor:
        inputs = processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        text_inputs = {
            k: v.to(device)
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask")
        }
        with torch.no_grad():
            z = model.get_text_features(**text_inputs).float()
            z = F.normalize(z, dim=-1)
        return z.cpu()

    return TextEncoder("siglip2", model_id, encode_batch)


def load_encoder(model_name: str, model_id: str, device: torch.device) -> TextEncoder:
    loaders = {
        "clip": load_clip_encoder,
        "dinotxt": load_dinotxt_encoder,
        "siglip2": load_siglip2_encoder,
    }
    return loaders[model_name](model_id, device)


def compute_year_embeddings(
    encoder: TextEncoder,
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
    batch_size: int = 32,
) -> tuple[List[int], torch.Tensor]:
    years = list(range(year_start, year_end + 1))
    sentences = year_sentences(year_start, year_end)
    embeds = []
    for start in su.log.tqdm_iterator(
        range(0, len(sentences), batch_size),
        desc=f"Computing {encoder.name} embeddings",
    ):
        batch = sentences[start : start + batch_size]
        embeds.append(encoder.encode_batch(batch))
    Z = torch.cat(embeds, dim=0)
    return years, Z


def visualize_year_embeddings_3d_pca(
    Z,
    years,
    title_suffix: str,
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
    ax.set_title(f"'In the year x' embeddings ({title_suffix})")
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


def default_paths(model_name: str, out_dir: str) -> Dict[str, str]:
    return {
        "embeds": os.path.join(out_dir, f"year_embeddings_{model_name}.pt"),
        "plot": os.path.join(out_dir, f"year_embeddings_3d_pca_{model_name}.pdf"),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and visualize year embeddings with CLIP, DINO.txt, and/or SigLIP2."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        choices=[*MODEL_CHOICES, "all"],
        help="Which encoders to run (default: all).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--year_start", type=int, default=YEAR_START)
    parser.add_argument("--year_end", type=int, default=YEAR_END)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--elev", type=float, default=22)
    parser.add_argument("--azim", type=float, default=-58)
    parser.add_argument(
        "--clip_model",
        type=str,
        default=MODEL_DEFAULTS["clip"],
        help="CLIP checkpoint id passed to clip.load.",
    )
    parser.add_argument(
        "--dinotxt_model",
        type=str,
        default=MODEL_DEFAULTS["dinotxt"],
        help="DINO.txt torch.hub model name.",
    )
    parser.add_argument(
        "--siglip2_model",
        type=str,
        default=MODEL_DEFAULTS["siglip2"],
        help="HuggingFace model id for SigLIP2.",
    )
    parser.add_argument(
        "--no_show",
        action="store_true",
        help="Do not display plots interactively.",
    )
    return parser.parse_args()


def resolve_models(models_arg: Sequence[str]) -> List[str]:
    if "all" in models_arg:
        return list(MODEL_CHOICES)
    return list(dict.fromkeys(models_arg))


def model_id_for(args, model_name: str) -> str:
    return {
        "clip": args.clip_model,
        "dinotxt": args.dinotxt_model,
        "siglip2": args.siglip2_model,
    }[model_name]


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("CUDA unavailable; running on CPU.")

    model_names = resolve_models(args.models)
    model_id_overrides = {name: model_id_for(args, name) for name in model_names}

    for model_name in model_names:
        paths = default_paths(model_name, args.out_dir)
        model_id = model_id_overrides[model_name]

        print(f"\n=== {model_name} ({model_id}) ===")
        encoder = load_encoder(model_name, model_id, device)
        years, Z = compute_year_embeddings(
            encoder,
            year_start=args.year_start,
            year_end=args.year_end,
            batch_size=args.batch_size,
        )
        print(f"Computed embeddings: {Z.shape}")

        torch.save(
            {
                "years": years,
                "embeddings": Z,
                "model": model_name,
                "model_id": model_id,
            },
            paths["embeds"],
        )
        print(f"Saved embeddings: {paths['embeds']}")

        visualize_year_embeddings_3d_pca(
            Z,
            years,
            title_suffix=f"{model_name}: {model_id}",
            out_path=paths["plot"],
            show=not args.no_show,
            elev=args.elev,
            azim=args.azim,
        )


if __name__ == "__main__":
    main()
