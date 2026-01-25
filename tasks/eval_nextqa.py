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

from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict as edict
import numpy as np
import json

import shared.utils as su
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)
from models.modeling_encoders import AutoEncoder
from notebooks.eval_care_retrieval import load_model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/work/piyush/pretrained_checkpoints/TARA")
    args = parser.parse_args()
    
    # Load model
    model_path = args.model_path
    vfc, tfc, vp = load_model(_id=model_path, device_map='auto', n_frames=16)
    
    # Load dataset
    data_dir = "/scratch/shared/beegfs/piyush/datasets/NExTQA"
    video_dir = f"{data_dir}/NExTVideo"
    df = pd.read_csv(f"{data_dir}/mc.csv")


    # Compute video embeddings
    norm = lambda x: torch.nn.functional.normalize(x, dim=-1)
    video_ids = df.video.unique()
    video_feat = {}
    for video_id in su.log.tqdm_iterator(video_ids, desc="Embedding videos"):
        video_path = f"{video_dir}/{video_id}.mp4"
        # assert os.path.exists(video_path)
        try:
            zv = norm(vfc(vp(video_path)))
            video_feat[video_id] = zv
        except:
            print(f"Failed for {video_id}.")
            continue
    import ipdb; ipdb.set_trace()