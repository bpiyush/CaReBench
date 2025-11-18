import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"

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


data_dir = "/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/egocvr"
clip_dir = f"{data_dir}/clips"
save_dir = f"{data_dir}/embs"

df_anno = pd.read_csv(f"{data_dir}/egocvr_annotations.csv")
df_base = pd.read_csv(f"{data_dir}/egocvr_data.csv")
df_gall = pd.read_csv(f"{data_dir}/egocvr_annotations_gallery.csv")

df_anno.shape, df_base.shape, df_gall.shape


from models.modeling_encoders import AutoEncoder
model_id = "/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/"\
    "nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint"
encoder = AutoEncoder.from_pretrained(model_id, device_map='auto')
su.misc.num_params(encoder.model)


# For re-ranking, we only need the video embeddings for queries
query_embeddings_video_only = {}
for i in su.log.tqdm_iterator(range(len(df_gall)), desc='Computing query embeddings'):
    row = df_gall.iloc[i].to_dict()
    query_video_path = f"{clip_dir}/{row['video_clip_id']}.mp4"
    try:
        video_tensor = read_frames_decord(query_video_path, num_frames=8)
        with torch.no_grad():
            zq = encoder.encode_vision(video_tensor.unsqueeze(0))
            zq = torch.nn.functional.normalize(zq, dim=-1).cpu().squeeze(0).float()
        query_embeddings_video_only[f"{row['video_clip_id']}"] = zq
    except:
        print(f"Error computing query embedding for {i}. Skipping.")
        continue
len(query_embeddings_video_only)
import ipdb; ipdb.set_trace()
torch.save(query_embeddings_video_only, f"{save_dir}/tara+tarsier_query_embeddings_video_only-nframes_8.pt")