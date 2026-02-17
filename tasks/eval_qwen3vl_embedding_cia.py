"""Evaluate Chiral Retrieval with Qwen3-VL-Embedding model."""
import os
import torch
import argparse

import shared.utils as su
from models.qwen3vl_embedding import Qwen3VLEmbedder
from tasks.eval_chiral_retrieval import load_data
from utils.chiral_retrieval_metrics import (
    compute_metrics,
    print_metrics_as_latex_row,
)


def pretty_print_args(args):
    print("========== Arguments ==========")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("================================\n")


def gather_text_features(df, model):
    """Compute text embeddings for all unique text IDs."""
    text_ids = df['text_id'].unique()
    texts_feat = {}
    for j, text_id in enumerate(
        su.log.tqdm_iterator(text_ids, desc='Computing text features')
    ):
        text = df[df.text_id == text_id].template.unique()[0]
        emb = model.process([{'text': text}])
        zt = emb.squeeze(0).cpu().float()
        texts_feat[text_id] = zt
        if j == 0:
            print("Text embedding:", zt.shape)
    return texts_feat


def gather_video_features(df, model, fps, max_frames):
    """Compute video embeddings for all unique video IDs."""
    video_ids = df.id.unique()
    video_feat = {}
    for j, video_id in enumerate(
        su.log.tqdm_iterator(video_ids, desc='Computing video features')
    ):
        video_path = df[df.id == video_id].video_path.unique()[0]
        emb = model.process([{
            'video': video_path,
            'fps': fps,
            'max_frames': max_frames,
        }])
        zv = emb.squeeze(0).cpu().float()
        video_feat[video_id] = zv
        if j == 0:
            print("Video embedding:", zv.shape)
    return video_feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model_name_or_path', type=str,
        default="/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B",
    )
    parser.add_argument('-d', '--dataset', type=str, default='ssv2')
    parser.add_argument('--device_map', type=str, default='cuda:0')
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--max_frames', type=int, default=16)
    parser.add_argument('--eval_on_subset', action='store_true')
    parser.add_argument('--no_save_embs', action='store_true')
    args = parser.parse_args()

    pretty_print_args(args)

    # Load model
    su.log.print_update(f"Loading Qwen3VLEmbedder from {args.model_name_or_path}")
    model = Qwen3VLEmbedder(
        model_name_or_path=args.model_name_or_path,
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
        device_map=args.device_map,
    )
    su.misc.num_params(model.model)

    # Load data
    df = load_data(dataset=args.dataset, split='validation')
    df = df.drop_duplicates(subset=['id', 'text_id']).reset_index(drop=True)

    if args.eval_on_subset:
        df = df.sample(frac=0.2, random_state=42).reset_index(drop=True)
        print(f"Evaluating on {len(df)} samples only.")
    else:
        print(f"Evaluating on all {len(df)} samples.")

    # Compute features
    texts_feat = gather_text_features(df, model)
    video_feat = gather_video_features(df, model, fps=args.fps, max_frames=args.max_frames)

    # Compute and print metrics
    metrics = compute_metrics(df, video_feat, texts_feat, show_metrics=True)
    print_metrics_as_latex_row(metrics)

    # Save embeddings
    save_embs = not args.no_save_embs
    if save_embs:
        save_dir = os.path.join(args.model_name_or_path, 'embs')
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving embeddings to {save_dir}")
        torch.save(video_feat, os.path.join(save_dir, f'video_feat-{args.dataset}.pt'))
        torch.save(texts_feat, os.path.join(save_dir, f'texts_feat-{args.dataset}.pt'))
