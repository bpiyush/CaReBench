import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"
from copy import deepcopy

import torch
import pandas as pd
import numpy as np
import decord
import json
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import PIL.Image
from glob import glob
from natsort import natsorted

import shared.utils as su
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)


def load_embeddings(file_prefix):
    embedding_files = glob(
        f"{save_dir}/{file_prefix}*.pt"
    )
    embeddings = {}
    for f in embedding_files:
        embeddings.update(torch.load(f))
    print("Number of embeddings: ", len(embeddings))
    return embeddings


if __name__ == "__main__":
    # Load data
    data_dir = "/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/egocvr"
    clip_dir = f"{data_dir}/clips"
    save_dir = f"{data_dir}/embs"
    df_anno = pd.read_csv(f"{data_dir}/egocvr_annotations.csv")
    df_base = pd.read_csv(f"{data_dir}/egocvr_data.csv")
    df_gall = pd.read_csv(f"{data_dir}/egocvr_annotations_gallery.csv")
    df_gall['query_id'] = df_gall[['video_clip_id', 'instruction']].apply(
        lambda x: f"{x[0]}|{x[1]}", axis=1,
    )

    # Load features
    query_embeddings_video_only = torch.load(
        f"{save_dir}/tara+tarsier_query_embeddings_video_only-nframes_8.pt"
    )
    global_gallery_embeddings = load_embeddings(
        "tarsier+tara_global_gallery_embeddings_egocvr",
    )
    local_gallery_embeddings = load_embeddings(
        "tarsier+tara_local_gallery_candidate_embeddings",
    )
    query_embeddings = torch.load(
        f"{save_dir}/tarsier+tara_query_embeddings_egocvr-frames_15.pt",
    )
    
    df_gall = df_gall[
        df_gall['query_id'].isin(set(query_embeddings_video_only.keys()))
    ].reset_index(drop=True)


    df_gall_reranked = []
    n_reranked = 15
    for query_id in su.log.tqdm_iterator(df_gall.query_id.unique(), desc='Re-ranking'):
        row_query = df_gall[df_gall.query_id == query_id].iloc[0].to_dict()
        zq = query_embeddings_video_only[query_id]

        # Local candidates
        local_idx = eval(row_query['local_idx'])
        rows_local = df_base.iloc[local_idx]
        zl = torch.stack([local_gallery_embeddings[x] for x in rows_local.clip_name])

        # Global candidates
        global_idx = eval(row_query['global_idx'])
        rows_global = df_base.iloc[global_idx]
        zg = torch.stack([global_gallery_embeddings[x] for x in rows_global.clip_name])
        
        sim_local = zq @ zl.T
        sim_global = zq @ zg.T
        
        reranked_indices_local = sim_local.topk(k=n_reranked).indices
        reranked_indices_global = sim_global.topk(k=n_reranked).indices
        local_idx_reranked = str(list(reranked_indices_local.cpu().numpy()))
        global_idx_reranked = str(list(reranked_indices_global.cpu().numpy()))

        row_query_new = deepcopy(row_query)
        row_query_new['local_idx'] = local_idx_reranked
        row_query_new['global_idx'] = global_idx_reranked
        df_gall_reranked.append(row_query_new)
    df_gall_reranked = pd.DataFrame(df_gall_reranked)
    import ipdb; ipdb.set_trace()