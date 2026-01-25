import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import decord
import json
import random
from IPython.display import display, Markdown, Latex
from itertools import permutations
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import PIL.Image
from glob import glob
from natsort import natsorted

from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict as edict
import numpy as np
import json
import pandas as pd
from collections import defaultdict
import ast
import torch
import numpy as np
from decord import VideoReader, cpu

import shared.utils as su
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)
from models.modeling_encoders import AutoEncoder
from notebooks.eval_care_retrieval import load_model


def create_video_dataframe(data: list) -> pd.DataFrame:
    """
    Convert a list of clip entries to a DataFrame where each row is a video.
    """
    video_clips = defaultdict(list)
    
    for clip in data:
        youtube_id = clip['youtube_id']
        clip_id = clip['id']  # e.g., "xHr8X2Wpmno_0"
        
        segment_str = clip['segment']
        segment = ast.literal_eval(segment_str.replace('. ', ', '))
        
        caption = clip.get('caption') or clip.get('sentence')
        
        video_clips[youtube_id].append({
            'clip_id': clip_id,
            'segment': segment,
            'caption': caption
        })
    
    rows = []
    for youtube_id, clips in video_clips.items():
        # Sort clips by start timestamp
        clips_sorted = sorted(clips, key=lambda x: x['segment'][0])
        
        clip_ids = [c['clip_id'] for c in clips_sorted]
        timestamps = [c['segment'] for c in clips_sorted]
        captions = [c['caption'] for c in clips_sorted]
        
        rows.append({
            'youtube_id': youtube_id,
            'clip_ids': clip_ids,
            'timestamps': timestamps,
            'captions': captions
        })
    
    return pd.DataFrame(rows)


def sample_segments(df: pd.DataFrame, n_samples: int = 4, seed: int = 42) -> pd.DataFrame:
    """
    Sample n non-overlapping segments from each video.
    """
    if seed is not None:
        random.seed(seed)
    
    rows = []
    for _, row in df.iterrows():
        clip_ids = row['clip_ids']
        timestamps = row['timestamps']
        captions = row['captions']
        
        if len(timestamps) < n_samples:
            continue
        
        indices = random.sample(range(len(timestamps)), n_samples)
        indices.sort()
        
        sampled_clip_ids = [clip_ids[i] for i in indices]
        sampled_timestamps = [timestamps[i] for i in indices]
        sampled_captions = [captions[i] for i in indices]
        
        rows.append({
            'youtube_id': row['youtube_id'],
            'clip_ids': sampled_clip_ids,
            'timestamps': sampled_timestamps,
            'captions': sampled_captions
        })
    
    return pd.DataFrame(rows)


def load_frames_from_row(
    row: dict,
    video_dir: str,
    n_frames_per_clip: int = 4,
) -> torch.Tensor:
    """
    Load frames from individual clip videos.
    
    Args:
        row: DataFrame row with 'clip_ids'
        video_dir: Base directory containing clip videos (e.g., "/path/to/test")
        n_frames_per_clip: Number of frames to sample per clip
    
    Returns:
        Tensor of shape (n_clips * n_frames_per_clip, H, W, C), dtype torch.uint8
    """
    clip_ids = row['clip_ids']
    
    all_frames = []
    
    for clip_id in clip_ids:
        # Construct clip video path
        video_path = glob(f"{video_dir}/{clip_id}.*")[0]
        # video_path = f"{video_dir}/{clip_id}.mp4"
        
        # Load video
        vr = VideoReader(video_path, ctx=cpu(0))
        n_total_frames = len(vr)
        
        # Sample n_frames_per_clip uniformly across the clip
        frame_indices = np.linspace(0, n_total_frames - 1, n_frames_per_clip, dtype=int)
        
        # Load frames
        frames = vr.get_batch(frame_indices.tolist())  # (n_frames, H, W, C)
        all_frames.append(frames)

    all_frames = torch.cat(all_frames, dim=0)
    
    return all_frames.to(torch.uint8)


def get_caption_permutations(captions: list) -> list:
    """
    Generate all permutations of captions, each joined by '; '.
    
    Args:
        captions: List of caption strings
    
    Returns:
        List of strings, each being a permutation of captions joined by '; '
    """
    return ['; '.join(perm) for perm in permutations(captions)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="/work/piyush/pretrained_checkpoints/Tarsier-7b")
    parser.add_argument('--n_frames_per_clip', type=int, default=4)
    parser.add_argument('--n_clips', type=int, default=4)
    args = parser.parse_args()
    
    data_dir = "/scratch/shared/beegfs/piyush/datasets/YouCook2"
    video_dir = f"{data_dir}/YouCookIIVideos"
    json_path = f"{data_dir}/val.json"
    data = su.io.load_json(json_path)
    
    df = create_video_dataframe(data)
    subdf = sample_segments(df, n_samples=args.n_clips)
    
    # Load model
    model_path = args.model_path
    n_frames = args.n_frames_per_clip * args.n_clips
    vfc, tfc, vp = load_model(_id=model_path, device_map='auto', n_frames=n_frames, attn_implementation="flash_attention_2")


    # Compute video and text embeddings for all permutations
    sims = []
    iterator = su.log.tqdm_iterator(range(len(subdf)))
    for i in iterator:
        row = subdf.iloc[i].to_dict()
        try:
            frames = load_frames_from_row(row, video_dir=f"{data_dir}/YouCookIIVideos/val", n_frames_per_clip=args.n_frames_per_clip)
            joint_captions = get_caption_permutations(row['captions'])

            # Compute video embeddings
            x = frames.permute((0, 3, 1, 2))
            zv = vfc(x)
            zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float()
            
            # Compute text embeddings for all permutations
            zt = torch.nn.functional.normalize(
                torch.stack([tfc(x) for x in joint_captions]), dim=-1
            ).cpu().float()
        except:
            continue
        sims.append(zv @ zt.T)
    sims = torch.stack(sims)
    
    
    from testoftime_eval.ranking_metrics import (
        compute_ranking_metrics,
        all_permutations,
    )
    from math import factorial
    K = 4
    M = factorial(K)
    perms = all_permutations(K)
    per_video_df, aggregate = compute_ranking_metrics(sims, n_captions=K, perms=perms, topk_for_recall=(1,3,5), ndcg_k=5)

    print(per_video_df.head())
    print("AGG:", aggregate)
