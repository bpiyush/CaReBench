#!/users/piyush/miniconda3/envs/qwen/bin/python
"""Compute text and image embeddings for the TIME10k / MatterOfTime dataset."""
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "False"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import argparse
import json
from typing import Dict, List, Sequence

import pandas as pd
import torch

import shared.utils as su
from mllm4emb.vlm_encoders import MODEL_DEFAULTS, load_encoder

DATASET_NAME = "time10k"
IMAGE_INDEX_CACHE = "valid_image_index.json"
IMAGE_INDEX_VERSION = 1

CLASS_LABELS = {
    "cars": "car",
    "ships": "ship",
    "aircrafts": "aircraft",
    "mobilephones": "mobile phone",
    "musicinstruments": "musical instrument",
    "weapons_and_ammo": "weapon",
}

MODEL_CHOICES = ("clip", "dinotxt", "siglip2")


def _stage(msg: str) -> None:
    print(msg, flush=True)


def image_index_cache_path(data_root: str) -> str:
    return os.path.join(data_root, IMAGE_INDEX_CACHE)


def _scan_image_index(image_dir: str) -> tuple[Dict[str, str], int]:
    from PIL import Image

    fnames = os.listdir(image_dir)
    _stage(f"Validating images in {image_dir} ({len(fnames)} files)...")
    index = {}
    skipped = 0
    for i, fname in enumerate(fnames, start=1):
        stem, _ = os.path.splitext(fname)
        path = os.path.join(image_dir, fname)
        try:
            with Image.open(path) as img:
                rgb = img.convert("RGB")
                rgb.load()
            index[stem] = path
        except Exception:
            skipped += 1
        if i % 1000 == 0 or i == len(fnames):
            _stage(f"  validated {i}/{len(fnames)} files ({len(index)} ok, {skipped} skipped)")
    if skipped:
        _stage(f"Skipped {skipped} non-image or unreadable files in {image_dir}")
    return index, skipped


def _save_image_index_cache(
    cache_path: str, image_dir: str, index: Dict[str, str], file_count: int, skipped: int
) -> None:
    payload = {
        "version": IMAGE_INDEX_VERSION,
        "image_dir": os.path.abspath(image_dir),
        "file_count": file_count,
        "skipped": skipped,
        "n_valid": len(index),
        "index": index,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    _stage(f"Wrote image index cache: {cache_path} ({len(index)} valid IDs)")


def _load_image_index_cache(cache_path: str, image_dir: str, file_count: int) -> Dict[str, str] | None:
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("version") != IMAGE_INDEX_VERSION:
        _stage(f"Image index cache version mismatch; will rescan.")
        return None
    if payload.get("image_dir") != os.path.abspath(image_dir):
        _stage("Image index cache path mismatch; will rescan.")
        return None
    if payload.get("file_count") != file_count:
        _stage(
            f"Image index cache stale (file count {payload.get('file_count')} != {file_count}); will rescan."
        )
        return None
    index = payload["index"]
    _stage(
        f"Loaded image index cache: {cache_path} "
        f"({len(index)} valid IDs, {payload.get('skipped', 0)} skipped previously)"
    )
    return index


def load_image_index(data_root: str, rebuild: bool = False) -> Dict[str, str]:
    image_dir = os.path.join(data_root, "images")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    file_count = len(os.listdir(image_dir))
    cache_path = image_index_cache_path(data_root)

    if not rebuild:
        cached = _load_image_index_cache(cache_path, image_dir, file_count)
        if cached is not None:
            return cached

    index, skipped = _scan_image_index(image_dir)
    _save_image_index_cache(cache_path, image_dir, index, file_count, skipped)
    return index


def load_dataset(data_root: str, csv_name: str = "time10k.csv", rebuild_image_index: bool = False):
    csv_path = os.path.join(data_root, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    _stage(f"Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    image_index = load_image_index(data_root, rebuild=rebuild_image_index)

    records = []
    for _, row in df.iterrows():
        image_path = image_index.get(str(row["id"]))
        if image_path is None:
            continue
        cls = str(row["class"])
        year = int(row["groundtruth_year"])
        label = CLASS_LABELS.get(cls, cls.replace("_", " "))
        text = f"An image of {label} in year {year}"
        records.append(
            {
                "id": str(row["id"]),
                "year": year,
                "class": cls,
                "object_name": str(row.get("object_name", "")),
                "text": text,
                "image_path": image_path,
            }
        )
    _stage(f"Loaded {len(records)} rows with images from {csv_path}")
    return records


def feature_path(out_dir: str, model_name: str, modality: str) -> str:
    return os.path.join(out_dir, f"{model_name}_{DATASET_NAME}_{modality}.pt")


def encode_modality(
    encoder,
    records: List[dict],
    modality: str,
    batch_size: int,
) -> torch.Tensor:
    embed_fn = (
        encoder.encode_text_batch
        if modality == "text"
        else encoder.encode_image_batch
    )
    key = "text" if modality == "text" else "image_path"
    embeds = []
    for start in su.log.tqdm_iterator(
        range(0, len(records), batch_size),
        desc=f"{encoder.name} {modality}",
    ):
        batch = records[start : start + batch_size]
        values = [row[key] for row in batch]
        embeds.append(embed_fn(values))
    return torch.cat(embeds, dim=0)


def save_features(path: str, records: List[dict], embeddings: torch.Tensor, meta: dict):
    payload = {
        **meta,
        "ids": [r["id"] for r in records],
        "years": [r["year"] for r in records],
        "classes": [r["class"] for r in records],
        "texts": [r["text"] for r in records],
        "image_paths": [r["image_path"] for r in records],
        "embeddings": embeddings.cpu().float(),
    }
    torch.save(payload, path)
    print(f"Saved {path}  shape={tuple(embeddings.shape)}")


def compute_model_features(
    model_name: str,
    model_id: str,
    records: List[dict],
    out_dir: str,
    device: torch.device,
    batch_size: int,
    modalities: Sequence[str],
    overwrite: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    pending = []
    for modality in modalities:
        path = feature_path(out_dir, model_name, modality)
        if os.path.exists(path) and not overwrite:
            _stage(f"Skipping existing file: {path}")
            continue
        pending.append(modality)
    if not pending:
        _stage(f"All requested {model_name} features already exist; nothing to do.")
        return

    _stage(
        f"Loading {model_name} encoder ({model_id}) on {device} "
        f"for modalities: {', '.join(pending)}..."
    )
    encoder = load_encoder(model_name, model_id, device)
    _stage(f"Encoder ready: {model_name}")
    meta_base = {
        "model": model_name,
        "model_id": model_id,
        "dataset": DATASET_NAME,
    }

    for modality in pending:
        path = feature_path(out_dir, model_name, modality)
        _stage(f"Encoding {model_name} {modality} ({len(records)} items, batch_size={batch_size})...")
        embeddings = encode_modality(encoder, records, modality, batch_size)
        save_features(
            path,
            records,
            embeddings,
            {**meta_base, "modality": modality},
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Compute TIME10k text/image VLM features.")
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/Time10K",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/work/piyush/experiments/MatterOfTime/features",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["clip"],
        choices=[*MODEL_CHOICES, "all"],
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["text", "image"],
        choices=["text", "image"],
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild_image_index",
        action="store_true",
        help="Rescan and revalidate all images instead of using valid_image_index.json cache.",
    )
    parser.add_argument("--clip_model", type=str, default=MODEL_DEFAULTS["clip"])
    parser.add_argument("--dinotxt_model", type=str, default=MODEL_DEFAULTS["dinotxt"])
    parser.add_argument("--siglip2_model", type=str, default=MODEL_DEFAULTS["siglip2"])
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
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    _stage(
        f"TIME10k feature extraction: models={resolve_models(args.models)}, "
        f"modalities={args.modalities}, device={device}, batch_size={args.batch_size}"
    )
    if device.type == "cpu":
        _stage("CUDA unavailable; running on CPU.")
    elif torch.cuda.is_available():
        _stage(f"CUDA device: {torch.cuda.get_device_name(device)}")

    records = load_dataset(args.data_root, rebuild_image_index=args.rebuild_image_index)
    model_names = resolve_models(args.models)

    for model_name in model_names:
        _stage(f"\n=== {model_name} ===")
        compute_model_features(
            model_name=model_name,
            model_id=model_id_for(args, model_name),
            records=records,
            out_dir=args.out_dir,
            device=device,
            batch_size=args.batch_size,
            modalities=args.modalities,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
