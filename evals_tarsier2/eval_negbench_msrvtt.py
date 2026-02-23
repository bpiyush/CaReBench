"""Evaluates retrieval on NegBench dataset."""
import os
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import json

import shared.utils as su


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/")
    parser.add_argument("--model_name", type=str, default="tarsier2_7b")
    args = parser.parse_args()


    from models.modeling_encoders import AutoEncoder
    model = AutoEncoder.from_pretrained(
        args.model_path,
        device_map="auto",
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)

    
    # Load data
    csv_path = "/scratch/shared/beegfs/piyush/datasets/NegBench/videos/msr_vtt_retrieval.csv"
    video_dir = "/scratch/shared/beegfs/piyush/datasets/MSRVTT/videos/all"
    df = pd.read_csv(csv_path)
    df['path'] = df['image_id'].apply(lambda x: os.path.join(video_dir, f"{x}.mp4"))
    df = df[df.path.apply(os.path.exists)]
    df['text'] = df['captions'].apply(lambda x: eval(x)[0])

    # Load negative retrieval CSV
    df_neg = pd.read_csv(
        "/scratch/shared/beegfs/piyush/datasets/NegBench/videos/msr_vtt_retrieval_rephrased_llama.csv"
    )
    df_neg['text'] = df_neg['captions'].apply(lambda x: eval(x)[0])
    
    # Compute text features: standard
    texts = df['text'].unique()
    texts_feat = {}
    for text in su.log.tqdm_iterator(texts, desc='Computing text features'):
        with torch.no_grad():
            zt = model.encode_text(text)
            zt = torch.nn.functional.normalize(zt, dim=-1).cpu().float().squeeze(0)
        texts_feat[text] = zt
    
    # Compute text features: negative
    texts_neg = df_neg['text'].unique()
    texts_neg_feat = {}
    for text in su.log.tqdm_iterator(texts_neg, desc='Computing text features'):
        with torch.no_grad():
            zt = model.encode_text(text)
            zt = torch.nn.functional.normalize(zt, dim=-1).cpu().float().squeeze(0)
        texts_neg_feat[text] = zt
    
    # Compute video features
    videos = df['path'].unique()
    videos_feat = {}
    for path in su.log.tqdm_iterator(videos, desc='Computing video features'):
        
        with torch.no_grad():
            zv = model.encode_vision(path)
            zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float().squeeze(0)
        videos_feat[path] = zv.cpu().float()
    
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
    
    # Save metrics
    metrics = {
        "standard": metrics_standard,
        "negative": metrics_negative,
    }
    save_dir = os.path.join(args.model_path, "metrics")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f"metrics_negbench_msrvtt_{args.model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
