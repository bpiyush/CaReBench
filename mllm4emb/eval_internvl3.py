import os
import sys
import argparse

import torch
import numpy as np
import pandas as pd

import shared.utils as su
from utils.video import read_frames_decord
from notebooks.eval_care_retrieval import (
    load_data,
    compute_metrics,
    print_metrics_as_latex_row,
)
import json

from models.internvl3 import AutoEncoder


def get_attn_implementation():
    """Return 'flash_attention_2' if supported, otherwise fall back to 'sdpa'."""
    try:
        from flash_attn import flash_attn_func
        dummy = torch.randn(1, 1, 1, 16, dtype=torch.bfloat16, device="cuda")
        flash_attn_func(dummy, dummy, dummy)
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def main(model_path: str, dataset: str):
    attn_impl = get_attn_implementation()
    print(f"Using attention implementation: {attn_impl}")

    encoder = AutoEncoder.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
    )

    # Sanity check
    frames = read_frames_decord(video_path='./assets/demo.mp4', num_frames=16)
    text = "This video features a man slicing tomatoes in the kitchen."
    vision_emb = encoder.encode_vision(frames.unsqueeze(0))
    text_emb = encoder.encode_text(text)
    print(f'Vision embedding shape: {vision_emb.shape}')
    print(f'Text embedding shape: {text_emb.shape}')

    # Load data
    df = load_data(dataset=dataset)
    df = df.drop_duplicates(subset=['id', 'text_id']).reset_index(drop=True)
    print(f"Dataset: {dataset}, shape: {df.shape}")

    # Compute text features
    text_ids = df['text_id'].unique()
    texts_feat = {}
    j = 0
    for text_id in su.log.tqdm_iterator(text_ids, desc='Computing text features'):
        text = df[df.text_id == text_id].template.unique()[0]
        zt = encoder.encode_text(text).squeeze(0)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        texts_feat[text_id] = zt.cpu().float()
        if j == 0:
            print("Text embedding: ", zt.shape)
        j += 1

    # Compute video features
    video_paths = df.video_path.unique()
    video_ids = df.id.unique()
    video_feat = {}
    j = 0
    for video_path in su.log.tqdm_iterator(video_paths, desc='Computing video features'):
        frames = read_frames_decord(video_path=video_path, num_frames=16).unsqueeze(0)
        zv = encoder.encode_vision(frames)[0]
        zv = torch.nn.functional.normalize(zv, dim=-1)
        video_feat[video_ids[j]] = zv.cpu().float()
        if j == 0:
            print("Video embedding: ", zv.shape)
        j += 1

    metrics = compute_metrics(df, video_feat, texts_feat, show_metrics=False)
    print_metrics_as_latex_row(metrics, sep='& ')

    # Save metrics
    save_dir = os.path.join(model_path, 'metrics')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'metrics-{dataset}.json'), 'w') as f:
        json.dump(metrics, f)

    # Save embeddings
    save_dir = os.path.join(model_path, 'embs')
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving embeddings to {save_dir}")
    torch.save(video_feat, os.path.join(save_dir, f'video_feat-{dataset}.pt'))
    torch.save(texts_feat, os.path.join(save_dir, f'texts_feat-{dataset}.pt'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="/work/piyush/pretrained_checkpoints/InternVL3-8B",
    )
    parser.add_argument(
        "--dataset", type=str, default="charades",
    )
    args = parser.parse_args()
    main(model_path=args.model_path, dataset=args.dataset)
