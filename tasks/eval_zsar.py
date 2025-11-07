"""Evaluates zero-shot action recognition."""
import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import json
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

from typing import Dict, Union, List, Any
import shared.utils as su
from notebooks.eval_care_retrieval import load_model


DATA_ROOT = "/scratch/shared/beegfs/piyush/datasets"
DATA_CONFIG = {
    "ucf101": {
        'video_dir': f"{DATA_ROOT}/UCF101/videos_mp4",
        'csv': f"{DATA_ROOT}/UCF101/metadata/test01.csv",
        'ext': 'mp4',
        'id_col': 'id',
        'target': 'class',
    },
    'hmdb51': {
        'video_dir': f"{DATA_ROOT}/HMDB51/videos",
        'csv': f"{DATA_ROOT}/HMDB51/metadata/test.csv",
        'ext': 'avi',
        'id_col': 'id',
        'target': 'class',
    },
    'epic': {
        'video_dir': f"{DATA_ROOT}/EPIC-Kitchens-100/cut_clips",
        'csv': f"{DATA_ROOT}/EPIC-Kitchens-100/epic-kitchens-100-annotations/ek100_validation_clean.csv",
        'ext': 'MP4',
        'id_col': 'id',
        'target': 'verb_noun',
    },
    'kinetics-verbs': {
        'video_dir': f"/datasets/KineticsClean/",
        'csv': f"{DATA_ROOT}/Kinetics400/metadata/val_vfc.csv",
        'ext': 'mp4',
        'id_col': 'id',
        'target': 'class',
    },
}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='ucf101',
        choices=DATA_CONFIG.keys(),
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    # model_path = "/work/piyush/pretrained_checkpoints/Tarsier-7b"
    model_path = args.model_path
    vfc, tfc, vp  = load_model(_id=model_path)
    
    # Load dataset
    dataset = args.dataset
    data_config = DATA_CONFIG[dataset]
    df = pd.read_csv(data_config['csv'])
    if args.dataset == 'epic':
        df['verb_noun'] = df[['verb', 'noun']].apply(lambda x: f"{x[0]} {x[1]}", axis=1)
    print(f"Number of rows in {dataset}: {len(df)}")
    df['path'] = df[data_config['id_col']].apply(lambda x: f"{data_config['video_dir']}/{x}.{data_config['ext']}")
    df = df[df.path.apply(os.path.exists)]
    print(f"Number of rows with video: {len(df)}")
    
    if args.debug:
        # Only pick first 200 rows
        np.random.seed(42)
        df = df.sample(200, random_state=42)
        print(f"Number of rows after sampling: {len(df)}")
    
    # 1. Compute embeddings for text descriptions of the classes
    classes = df[data_config['target']].unique()
    text_embs = {}
    for text in su.log.tqdm_iterator(classes, desc='Computing text embeddings'):
        zt = tfc(text)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        text_embs[text] = zt.cpu().float()
    
    # 2. Compute embeddings for the videos
    vid_embs = {}
    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing video embeddings'):
        row = df.iloc[i].to_dict()
        
        try:
            zv = vfc(vp(row['path']))
            zv = torch.nn.functional.normalize(zv, dim=-1)
            vid_embs[row[data_config['id_col']]] = zv.cpu().float()
        except:
            print(f"Error computing video embedding for {row['path']}")
            continue

    # Only keep the rows with valid video embeddings
    df = df[df[data_config['id_col']].isin(list(vid_embs.keys()))]
    print(f"Number of rows with valid video embeddings: {len(df)}")
    
    zv = torch.stack([vid_embs[df.iloc[i].to_dict()[data_config['id_col']]] for i in range(len(df))])
    zt = torch.stack([text_embs[df.iloc[i].to_dict()[data_config['target']]] for i in range(len(df))])
    print("Video embeddings shape: ", zv.shape)
    print("Text embeddings shape: ", zt.shape)
    sim = zv @ zt.t()
    
    # Get predicted classes
    pred_indices = sim.argmax(dim=-1)
    true_classes = df[data_config['target']].tolist()
    pred_classes = [true_classes[i] for i in pred_indices]
    accuracy = np.mean([pred_classes[i] == true_classes[i] for i in range(len(df))])
    print(f"Accuracy: {accuracy:.2f}")
