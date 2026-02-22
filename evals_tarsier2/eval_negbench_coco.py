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
from tasks.eval_negbench_msrvtt import batchify, recall_at_k


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


def gather_text_embeddings(df, index=0):
    texts_feat = []
    for i in su.log.tqdm_iterator(range(len(df)), desc=f'Computing text features {index}'):
        text = eval(df.iloc[i].captions)[index]
        with torch.no_grad():
            zt = model.encode_text(text)
        zt = torch.nn.functional.normalize(zt, dim=-1).cpu().float()
        texts_feat.append(zt)
    texts_feat = torch.cat(texts_feat)
    return texts_feat


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
    data_dir = "/scratch/shared/beegfs/piyush/datasets/NegBench"
    image_dir = "/scratch/shared/beegfs/piyush/datasets/COCO2017"
    csv_name_std = "images/COCO_val_retrieval.csv"
    df_std = pd.read_csv(f"{data_dir}/{csv_name_std}")
    csv_name_neg = "images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
    df_neg = pd.read_csv(f"{data_dir}/{csv_name_neg}")


    image_feats = []
    for i in su.log.tqdm_iterator(range(len(df_std))):
        row = df_std.iloc[i].to_dict()
        image_path = row['filepath'].replace('data/coco/images', image_dir)
        with torch.no_grad():
            zi = model.encode_image(image_path)
            zi = torch.nn.functional.normalize(zi, dim=-1)
            zi = zi.cpu().float()
        image_feats.append(zi)
    image_feats = torch.cat(image_feats)
    image_feat = image_feats


    # Gather text embeddings
    texts_feat_std_all = []
    for j in range(5):
        texts_feat_std_all.append(gather_text_embeddings(df_std, j))
    texts_feat_std_all = torch.stack(texts_feat_std_all)
    
    
    texts_feat_neg_all = []
    for j in range(5):
        texts_feat_neg_all.append(gather_text_embeddings(df_neg, j))
    texts_feat_neg_all = torch.stack(texts_feat_neg_all)


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
    result_dir = os.path.join(args.model_path, "metrics")
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"metrics_negbench_coco_{args.model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)
