"""Evaluates retrieval on NegBench dataset."""
import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import json
from torch.nn.functional import cosine_similarity
from utils.video import read_frames_decord

import shared.utils as su
from notebooks.eval_care_retrieval import load_model


def read_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m', '--model_path', type=str,
        default="/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint/",
    )
    parser.add_argument('--device_map', type=str, default='auto')
    parser.add_argument(
        '-c', '--csv_path', type=str,
        default='/scratch/shared/beegfs/piyush/datasets/NegBench/videos/msr_vtt_retrieval.csv',
    )
    parser.add_argument(
        '-v', '--video_dir', type=str,
        default='/scratch/shared/beegfs/piyush/datasets/MSRVTT/videos/all',
    )
    args = parser.parse_args()
    return args


def recall_at_k(scores, positive_pairs, k):
    """
    Computes recall@k for a given set of scores and positive pairs.
    Args:
        scores: torch.Tensor
            The scores of the model.
        positive_pairs: torch.Tensor
            A binary tensor indicating positive pairs.
        k: int
            The value of k for recall@k.
    Returns:
        recall_at_k: torch.Tensor
            The recall@k value.
    """
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    """
    Applies a function to batches of data.
    Args:
        func: callable
            The function to apply.
        X: torch.Tensor
            The input data.
        Y: torch.Tensor
            The target data.
        batch_size: int
            The batch size.
        device: torch.device
            The device to use.
        *args: list
            Additional positional arguments to pass to func.
        **kwargs: dict
            Additional keyword arguments to pass to func.
    Returns:
        results: torch.Tensor
            The results of applying func to the data.
    """
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def compute_metrics(images_emb, texts_emb, df):
    scores = texts_emb @ images_emb.t()
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    texts_image_index = [
        np.where(df['text'] == text)[0][0] for text in df['text']
    ]
    texts_image_index = np.array(texts_image_index)
    positive_pairs[torch.arange(len(scores)), texts_image_index] = True
    
    # Compute the recall@k
    metrics = {}
    recall_k_list=[5]
    for recall_k in recall_k_list:
        metrics[f"image_retrieval_recall@{recall_k}"] = \
            (batchify(recall_at_k, scores, positive_pairs, 32, 'cpu', k=recall_k)>0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = \
            (batchify(recall_at_k, scores.T, positive_pairs.T, 32, 'cpu', k=recall_k)>0).float().mean().item()
    return metrics


if __name__ == "__main__":
    args = read_args()
    print(args)
    
    vfc, tfc, vp = load_model(_id=args.model_path, device_map=args.device_map)
    
    # Load data
    df = pd.read_csv(args.csv_path)
    df['path'] = df['image_id'].apply(lambda x: os.path.join(args.video_dir, f"{x}.mp4"))
    df = df[df.path.apply(os.path.exists)]
    df['text'] = df['captions'].apply(lambda x: eval(x)[0])

    # Load negative retrieval CSV
    df_neg = pd.read_csv(
        "/scratch/shared/beegfs/piyush/datasets/NegBench/videos/msr_vtt_retrieval_rephrased_llama.csv"
    )
    df_neg['text'] = df_neg['captions'].apply(lambda x: eval(x)[0])
    
    # Compute text features
    texts = df['text'].unique()
    texts_feat = {}
    for text in su.log.tqdm_iterator(texts, desc='Computing text features'):
        zt = tfc(text)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        texts_feat[text] = zt.cpu().float()
    
    texts_neg = df_neg['text'].unique()
    texts_neg_feat = {}
    for text in su.log.tqdm_iterator(texts_neg, desc='Computing text features'):
        zt = tfc(text)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        texts_neg_feat[text] = zt.cpu().float()
    
    # Compute video features
    videos = df['path'].unique()
    videos_feat = {}
    for path in su.log.tqdm_iterator(videos, desc='Computing video features'):
        video_tensor = vp(path)
        zv = vfc(video_tensor)
        zv = torch.nn.functional.normalize(zv, dim=-1)
        videos_feat[path] = zv.cpu().float()
    import ipdb; ipdb.set_trace()
    
    images_emb = torch.stack([videos_feat[path] for path in df['path']])
    texts_emb = torch.stack([texts_feat[text] for text in df['text']])
    texts_neg_emb = torch.stack([texts_neg_feat[text] for text in df_neg['text']])

    # Standard retrieval
    su.log.print_update("Standard retrieval")
    metrics_standard = compute_metrics(images_emb, texts_emb, df)
    print(metrics_standard)
    
    # Negative retrieval
    su.log.print_update("Negative retrieval")
    metrics_negative = compute_metrics(images_emb, texts_neg_emb, df_neg)
    print(metrics_negative)
