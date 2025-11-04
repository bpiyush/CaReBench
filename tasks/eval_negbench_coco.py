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
import PIL, PIL.Image
import einops

import shared.utils as su
from notebooks.eval_care_retrieval import load_model
from tasks.eval_negbench_msrvtt import *


def gather_text_embeddings(df, index=0):
    texts_feat = []
    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing text features'):
        text = eval(df.iloc[i].captions)[index]
        zt = tfc(text)
        zt = torch.nn.functional.normalize(zt, dim=-1).cpu().float()
        texts_feat.append(zt)
    texts_feat = torch.stack(texts_feat)
    return texts_feat


def gather_text_embeddings_batch(df, index=0, batch_size=16):
    texts_feat = []
    indices = np.arange(0, len(df), batch_size)
    for s in su.log.tqdm_iterator(indices, desc='Computing text features'):
        e = min(len(df), s + batch_size)
        vals = df.iloc[s:e].captions.tolist()
        text = [eval(x)[index] for x in vals]
        zt = tfc(text)
        zt = torch.nn.functional.normalize(zt, dim=-1).cpu().float()
        texts_feat.append(zt)
    texts_feat = torch.cat(texts_feat)
    return texts_feat


def read_image(image_path):
    image = PIL.Image.open(image_path).convert('RGB')
    image = torch.from_numpy(np.asarray(image))
    image = image.permute(2, 0, 1)  # (T, C, H, W), torch.uint8
    return image


def compute_image_embedding(image_path):
    image_tensor = read_image(image_path)
    with torch.no_grad():
        zi = vfc.encoder.encode_vision(image_tensor.unsqueeze(0)).cpu().squeeze(0).float()
        zi = torch.nn.functional.normalize(zi, dim=-1)
    return zi


def gather_image_embs(df_std):
    # image_feats = {}
    image_feats = []
    for i in su.log.tqdm_iterator(range(len(df_std))):
        row = df_std.iloc[i].to_dict()
        image_path = row['filepath'].replace('data/coco/images', image_dir)
        zi = compute_image_embedding(image_path)
        # image_feats[image_path] = zi
        image_feats.append(zi)
    image_feats = torch.stack(image_feats)
    return image_feats


def compute_metrics(images_emb, texts_emb, text_to_image_index_std):
    scores = texts_emb @ images_emb.t()
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), text_to_image_index_std] = True
    
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

    # Load model
    # model_path = "/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint/"
    # model_name = "tarsier7b+tara"
    model_path = "/work/piyush/pretrained_checkpoints/Tarsier-7b/"
    model_name = "tarsier7b"
    vfc, tfc, vp = load_model(_id=model_path, device_map="auto")
    
    
    # Load data
    data_dir = "/scratch/shared/beegfs/piyush/datasets/NegBench"
    image_dir = "/scratch/shared/beegfs/piyush/datasets/COCO2017"
    csv_name_std = "images/COCO_val_retrieval.csv"
    df_std = pd.read_csv(f"{data_dir}/{csv_name_std}")
    csv_name_neg = "images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
    df_neg = pd.read_csv(f"{data_dir}/{csv_name_neg}")


    # Gather text embeddings
    texts_feat_std_all = []
    texts_feat_neg_all = []
    for j in range(5):
        texts_feat_std_all.append(gather_text_embeddings_batch(df_std, j))
        texts_feat_neg_all.append(gather_text_embeddings_batch(df_neg, j))
    texts_feat_std_all = torch.stack(texts_feat_std_all)
    texts_feat_neg_all = torch.stack(texts_feat_neg_all)
    
    # Gather image embeddings
    image_feat = gather_image_embs(df_std)
    
    # Compute metrics
    text_to_image_index_std = np.arange(len(image_feat))
    text_std = einops.rearrange(texts_feat_std_all, 'j l d -> (j l) d')
    text_neg = einops.rearrange(texts_feat_neg_all, 'j l d -> (j l) d')
    text_to_image_index = np.arange(len(image_feat))
    text_to_image_index = np.concatenate([text_to_image_index] * 5)
    metrics = {
        'std': compute_metrics(image_feat, text_std, text_to_image_index),
        'neg': compute_metrics(image_feat, text_neg, text_to_image_index),
    }
    print(json.dumps(metrics, indent=4))
    
    # Save metrics
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"metrics_negbench_coco_{model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
