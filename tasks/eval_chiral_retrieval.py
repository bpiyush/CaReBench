"""Evaluate Chiral Retrieval with CaRe like models."""
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
from utils.chiral_retrieval_metrics import (
    compute_metrics, print_metrics_as_latex_row,
)


# Constants
DATA_ROOT = "/scratch/shared/beegfs/piyush/datasets"
VIDEO_DIR = {
    "ssv2": f"{DATA_ROOT}/SSv2/20bn-something-something-v2",
    "epic": f"{DATA_ROOT}/EPIC-Kitchens-100/cut_clips",
    "charades": f"{DATA_ROOT}/Charades/Charades_v1_480_cut_clips"
}
EXT = {
    'ssv2': 'webm',
    'epic': 'MP4',
    'charades': 'mp4',
}
REPO_PATH = os.path.expanduser("~/projects/TimeBound.v1/")


def pretty_print_args(args):
    print("========== Arguments ==========")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("================================\n")


def load_data(dataset='ssv2', split='validation'):
    su.log.print_update(f"Loading data for dataset: {dataset} and split: {split}")

    split_dir = f"{REPO_PATH}/adapt4change/chirality_in_action_splits"
    csv_path = f"{split_dir}/cia-{dataset}-{split}.csv"
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)

    # Add text ID
    df['text_id'] = df[['chiral_triplet_id', 'chiral_label']].apply(
        lambda x: f"{x[0]}_{x[1]}", axis=1,
    )
    video_dir = VIDEO_DIR[dataset]
    ext = EXT[dataset]

    df['video_path'] = df['id'].apply(lambda x: f"{video_dir}/{x}.{ext}")
    df = df[df.video_path.apply(os.path.exists)]
    print("Number of rows: ", len(df))
    print("Sample row: ")
    print(json.dumps(df.iloc[0].to_dict(), indent=4))
    su.log.print_update(f".")
    
    return df


# Define a video processor: video_path -> video_tensor
class VideoProcessor:
    def __init__(self, n_frames=16):
        self.n_frames = n_frames
    
    def __call__(self, video_path):
        video = read_frames_decord(video_path, self.n_frames)
        return video


# Define a feature computer: video_tensor -> video_feature
class VideoFeatureComputer:
    def __init__(self, encoder):
        self.encoder = encoder
    
    def __call__(self, video_tensor):
        with torch.no_grad():
            vision_emb = self.encoder.encode_vision(
                video_tensor.unsqueeze(0),
            ).cpu().squeeze(0).float()
        return vision_emb


# Define a text feature computer: text_str -> text_feature
class TextFeatureComputer:
    def __init__(self, encoder):
        self.encoder = encoder
    
    def __call__(self, text_str):
        with torch.no_grad():
            text_emb = self.encoder.encode_text(text_str).cpu().squeeze(0).float()
        return text_emb


def gather_video_features(df, vfc, vp):
    video_ids = df.id.unique()
    video_feat = {}
    j = 0
    for video_id in su.log.tqdm_iterator(video_ids, desc='Computing video features'):
        video_path = df[df.id == video_id].video_path.unique()[0]
        video_tensor = vp(video_path)
        zv = vfc(video_tensor)
        zv = torch.nn.functional.normalize(zv, dim=-1).float().cpu()
        video_feat[video_id] = zv
        j += 1
    return video_feat


def gather_text_features(df, tfc):
    # Compute text features
    text_ids = df['text_id'].unique()
    texts_feat = {}
    for text_id in su.log.tqdm_iterator(text_ids, desc='Computing text features'):
        text = df[df.text_id == text_id].template.unique()[0]
        zt = tfc(text)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        texts_feat[text_id] = zt.cpu().float()
    return texts_feat


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name_or_path', type=str, default=None)
    parser.add_argument('-d', '--dataset', type=str, default='ssv2')
    args = parser.parse_args()
    
    pretty_print_args(args)
    
    # Load data
    df = load_data(dataset=args.dataset, split='validation')
    
    
    # Load model
    from models.qwen3vl import AutoEncoder
    encoder = AutoEncoder.from_pretrained(
        args.model_name_or_path,
        device_map='auto',
        dtype=torch.bfloat16,
        attn_implementation='flash_attention_2',
    )
    su.misc.num_params(encoder.model)
    vp = VideoProcessor(n_frames=16)
    vfc = VideoFeatureComputer(encoder)
    tfc = TextFeatureComputer(encoder)
    
    video_feat = gather_video_features(df, vfc, vp)
    text_feat = gather_text_features(df, tfc)
    
    metrics = compute_metrics(df, video_feat, text_feat, show_metrics=True)
    print_metrics_as_latex_row(metrics)
