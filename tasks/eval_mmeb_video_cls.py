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


data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
meta_config = su.io.load_yml(
    '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
)

# Compute video embeddings
df_video = []
for ds_key in su.log.tqdm_iterator(meta_config, desc='Gathering video paths'):
    file_name = meta_config[ds_key]['json_name']
    data_file = f'{data_root}/video-tasks/data/{file_name}'
    assert os.path.exists(data_file)
    data = su.io.load_jsonl(data_file)

    ds_name = os.path.basename(meta_config[ds_key]['frame_root'])
    for d in data:
        video_id = d['video_id']
        video_dir = f"{data_root}/video-tasks/frames/{ds_name}/{video_id}"
        assert os.path.isdir(video_dir)
        df_video.append(
            dict(ds_key=ds_key, ds_name=ds_name, video_id=video_id, video_dir=video_dir)
        )
df_video = pd.DataFrame(df_video)
assert len(df_video.video_id.unique()) == len(df_video)

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
save_name = f"{model_name}_video_embeddings_mmebv2_video_cls.pt"
torch.save(video_embeddings, os.path.join(save_dir, save_name))