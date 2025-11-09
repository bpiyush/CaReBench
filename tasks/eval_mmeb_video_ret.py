import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"
from glob import glob

import torch
import pandas as pd
import numpy as np
import json
from torch.nn.functional import cosine_similarity
from utils.video import read_frames_decord
from IPython.display import display, Markdown, Latex

import shared.utils as su
from notebooks.eval_care_retrieval import load_model
from models.modeling_encoders import AutoEncoder
from glob import glob 
from natsort import natsorted
import PIL.Image


def read_frames_from_files(
        frame_dir, n_frames=16, sample='middle', fix_start=None, 
        max_num_frames=-1, trimmed30=False, height=-1, width=-1
    ):

    total_frames = len(os.listdir(frame_dir))
    indices = np.linspace(0, total_frames, n_frames, dtype=int, endpoint=False)

    files = np.array(natsorted(glob(f"{frame_dir}/*.*")))[indices]
    frames = [PIL.Image.open(f).convert("RGB") for f in files]
    frames = torch.from_numpy(np.stack([np.asarray(x) for x in frames]))
    frames = frames.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return frames


data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
meta_config = su.io.load_yml(
    '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml'
)
video_root = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames/data/ziyan/video_retrieval"

# This defines the huggingface repo and subset for each dataset
# (repo, subset, split)
json_paths = {
    "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
    "MSVD": ("VLM2Vec/MSVD", None, "test"),
    "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
    # "YouCook2": ("VLM2Vec/YouCook2", None, "val"), # HF version compatibility issue
    "YouCook2": ("lmms-lab/YouCook2", None, "val"),
    "VATEX": ("VLM2Vec/VATEX", None, "test"),
}

video_id_extractor = {
    "MSR-VTT": lambda x: x['video_id'],
    "MSVD": lambda x: x['video_id'],
    "DiDeMo": lambda x: x['video'].split('/')[-1].split('.')[0],
    "YouCook2": lambda x: x['id'],
    "VATEX": lambda x: x['videoID'],
}


df_video = {'video_id': [], 'video_dir': []}
from datasets import load_dataset
for ds_key in su.log.tqdm_iterator(meta_config, desc='Processing datasets'):
    print(ds_key)
    repo, subset, split = json_paths[ds_key]
    df = pd.DataFrame(load_dataset(repo, subset)[split])
    video_dir = f"{video_root}/{ds_key}/frames"
    video_ids = os.listdir(video_dir)
    assert len(video_ids) == len(df)
    # print(df.iloc[0])
    
    df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)
    df['video_dir'] = df['video_id'].apply(lambda x: f"{video_root}/{ds_key}/frames/{x}")
    df_video['video_id'].extend(df['video_id'].tolist())
    df_video['video_dir'].extend(df['video_dir'].tolist())
    print('-' * 100)
df_video = pd.DataFrame(df_video)
assert len(df_video.video_id.unique()) == len(df_video)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint')
parser.add_argument('--model_name', type=str, default='tarsier7b+tara')
args = parser.parse_args()

# Load model
n_frames = 8
model_id = args.model_id
model_name = args.model_name
encoder = AutoEncoder.from_pretrained(model_id, device_map='auto')
su.misc.num_params(encoder.model)


video_embeddings = {}
for i in su.log.tqdm_iterator(range(len(df_video)), desc='Computing video embeddings'):
    row = df_video.iloc[i].to_dict()
    try:
        video_tensor = read_frames_from_files(row['video_dir'], n_frames=n_frames)
        with torch.no_grad():
            zv = encoder.encode_vision(video_tensor.unsqueeze(0)).cpu().squeeze(0).float()
            zv = torch.nn.functional.normalize(zv, dim=-1)
        video_embeddings[row['video_id']] = zv
    except:
        print(f"Error computing video embedding for {row['video_id']}")
        continue
len(video_embeddings)
save_dir = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/features"
save_name = f"{model_name}_video_embeddings_mmebv2_video_ret.pt"
torch.save(video_embeddings, os.path.join(save_dir, save_name))