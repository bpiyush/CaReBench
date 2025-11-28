"""Computes TARA features on CMD."""
import os
import sys
import argparse
from glob import glob
from IPython.display import display, Markdown, Latex

import torch
import pandas as pd
import numpy as np
import decord
import shared.utils as su

from models.modeling_encoders import AutoEncoder
from utils.video import read_frames_decord


def embed_video(video_path, encoder, n_frames):
    video_tensor = read_frames_decord(video_path, n_frames)
    with torch.no_grad():
        zv = encoder.encode_vision(video_tensor.unsqueeze(0))
        zv = torch.nn.functional.normalize(zv, dim=-1).cpu().squeeze(0).float()
    return zv


def generate_shot_filename(video_id, start_time, end_time):
    """
    Generate a unique filename for the shot.
    Uses a hash of the video_id to keep filename manageable.
    """
    import hashlib
    video_hash = hashlib.md5(video_id.encode()).hexdigest()[:12]
    return f"{video_hash}_{start_time:.2f}_{end_time:.2f}.mp4"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute TARA features on CMD")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/CondensedMovies",
        help="Directory containing video files"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/"
                "nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint",
        help="Path to the captioner model"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=8,
        help="Number of frames to sample from each video"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=50000,
        help="Save checkpoint after processing this many videos"
    )
    parser.add_argument(
        "--si",
        type=int,
        default=None,
        help="Start index for video processing (after sorting)"
    )
    parser.add_argument(
        "--ei",
        type=int,
        default=None,
        help="End index for video processing (after sorting)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode with visual output"
    )
    
    args = parser.parse_args()
    
    # Load model
    def check_ampere_gpu():
        gpu_name = torch.cuda.get_device_name()
        if gpu_name in ['NVIDIA RTX A6000', 'NVIDIA RTX A5000', 'NVIDIA RTX A4000']:
            return True
        else:
            return False
    if check_ampere_gpu():
        attn_implementation = "flash_attention_2"
    else:
        attn_implementation = "sdpa"
    print(f"Using attn_implementation: {attn_implementation}")
    encoder = AutoEncoder.from_pretrained(args.model_id, device_map='auto', attn_implementation=attn_implementation)
    su.misc.num_params(encoder.model)
    
    # Test on a random video
    if args.verbose:
        zv = embed_video('./assets/demo.mp4', encoder, args.n_frames)
        print("Video embedding shape:", zv.shape)
    
    # Load CSV
    df = pd.read_csv(f"{args.data_dir}/metadata/shots.csv")
    print(f"Loaded {len(df)} rows from CSV.")
    
    # Filter by start and end index
    df = df.iloc[args.si:args.ei]
    df['id'] = df[['video_id', 'st', 'et']].apply(lambda x: generate_shot_filename(*x), axis=1)
    df['video_path'] = df['id'].apply(lambda x: f"{args.data_dir}/shots/{x}")
    print(f"Filtered {len(df)} rows from CSV.")
    
    # Filter out non existing videos
    from tqdm import tqdm
    tqdm.pandas(desc="Filtering out non existing videos")
    df = df[df['video_path'].progress_apply(os.path.exists)]
    print(f"Filtered {len(df)} rows from CSV.")
    
    # Save directory
    save_dir = f"{args.data_dir}/outputs/features-tara7b-n={args.n_frames}/"
    os.makedirs(save_dir, exist_ok=True)
    
    feats = {}
    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing features'):
        row = df.iloc[i].to_dict()
        try:
            zv = embed_video(row['video_path'], encoder, args.n_frames)
            feats[row['id']] = zv
        except Exception as e:
            print(f"Error computing feature for {row['id']}. Skipping.")
            print(f"Error: {e}")
            continue
    save_path = f"{save_dir}/{args.si}-{args.ei}.pt"
    torch.save(feats, save_path)
    print(f"Features (num={len(feats)}) saved to {save_path}.")