"""
Compute video + text embeddings for MSRVTT test-1K with a Tarsier2 / TARA encoder.

Saves a single dict to {model_path}/embs/{model_name}_msrvtt_test1k_embeddings.pt
with keys = video_id (e.g. video7020) and caption strings, matching the format
used by measure_modgap_over_training.py / plot_retrieval_vs_modgap.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure repo root is importable when launched via absolute path / tmux.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd
import torch

import shared.utils as su


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Checkpoint dir (TARA or Tarsier2-7b-0115).",
    )
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument(
        "--data_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/MSRVTT",
    )
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--device_map", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    data_root = args.data_root
    video_dir = os.path.join(data_root, "videos", "all")
    ann_path = os.path.join(data_root, "annotation", "msrvtt_test_1k.json")

    data = json.load(open(ann_path, "r", encoding="utf-8"))
    df = pd.DataFrame(data)
    df["video_path"] = df["video"].apply(lambda x: os.path.join(video_dir, x))
    assert df["video_path"].apply(os.path.exists).all(), "Some videos are missing"

    save_dir = os.path.join(args.model_path, "embs")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.model_name}_msrvtt_test1k_embeddings.pt")

    if os.path.exists(save_path):
        embeddings = torch.load(save_path, map_location="cpu", weights_only=False)
        print(f"Resuming from {save_path} ({len(embeddings)} keys)")
    else:
        embeddings = {}

    from models.modeling_encoders import AutoEncoder

    model = AutoEncoder.from_pretrained(
        args.model_path,
        device_map=args.device_map,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)

    # Videos
    for i in su.log.tqdm_iterator(range(len(df)), desc="Videos"):
        row = df.iloc[i]
        vid = str(row["video_id"])
        if vid in embeddings:
            continue
        try:
            with torch.inference_mode():
                z = model.encode_vision(row["video_path"]).squeeze(0).cpu().float()
            embeddings[vid] = torch.nn.functional.normalize(z, dim=-1)
        except Exception as e:
            print(f"Error encoding video {vid}: {e}")
            continue
        if (i + 1) % args.save_every == 0:
            torch.save(embeddings, save_path)
            print(f"Checkpointed {len(embeddings)} keys -> {save_path}")

    # Texts (unique captions)
    captions = list(dict.fromkeys(df["caption"].astype(str).tolist()))
    for i in su.log.tqdm_iterator(range(len(captions)), desc="Texts"):
        cap = captions[i]
        if cap in embeddings:
            continue
        try:
            with torch.inference_mode():
                z = model.encode_text(cap).squeeze(0).cpu().float()
            embeddings[cap] = torch.nn.functional.normalize(z, dim=-1)
        except Exception as e:
            print(f"Error encoding text [{cap[:60]}...]: {e}")
            continue
        if (i + 1) % args.save_every == 0:
            torch.save(embeddings, save_path)
            print(f"Checkpointed {len(embeddings)} keys -> {save_path}")

    torch.save(embeddings, save_path)
    n_vid = sum(1 for k in embeddings if str(k).startswith("video"))
    n_txt = len(embeddings) - n_vid
    print(f"Saved {len(embeddings)} embeddings ({n_vid} video, {n_txt} text) -> {save_path}")


if __name__ == "__main__":
    main()
