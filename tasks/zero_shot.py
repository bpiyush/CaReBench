import os
import json
from glob import glob

import torch
import pandas as pd
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

import shared.utils as su
from notebooks.eval_care_retrieval import load_model


if __name__ == "__main__":
    data_dir = "/scratch/shared/beegfs/piyush/datasets/Kinetics400"
    verb_dir = f"{data_dir}/verbs_in_action"
    classes = su.io.load_txt(f"{verb_dir}/classes.txt")

    # model_path = "/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint"
    model_path = "/work/piyush/pretrained_checkpoints/Tarsier-7b"
    vfc, tfc, vp  = load_model(_id=model_path)
    
    text_embs = {}
    for text in su.log.tqdm_iterator(classes, desc='Computing class embeddings'):
        zt = tfc(text)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        text_embs[text] = zt.cpu().float()
    
    df = pd.read_csv(f"{data_dir}/metadata/val.csv")
    df = df[df['class'].isin(classes)]
    video_dir = "/datasets/KineticsClean/"
    df['path'] = df.id.apply(lambda x: f"{video_dir}/{x}.mp4")
    df = df[df.path.apply(os.path.exists)]
    vid_embs = {}
    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing video embeddings'):
        row = df.iloc[i].to_dict()
        
        try:
            zv = vfc(vp(row['path']))
            zv = torch.nn.functional.normalize(zv, dim=-1)
            vid_embs[row['id']] = zv
        except:
            print(f"Error computing video embedding for {row['path']}")
            continue
    
    # Only keep the rows with valid video embeddings
    df = df[df.id.isin(list(vid_embs.keys()))]
    print(f"Number of rows with valid video embeddings: {len(df)}")

    ZT = torch.stack([text_embs[text] for text in classes])
    ZV = torch.stack([vid_embs[id] for id in df.id.tolist()])
    scores = ZV @ ZT.t()
    print(scores.shape)
    
    # Gather predictions for each video
    predictions = {}
    for i in su.log.tqdm_iterator(range(len(df)), desc='Gathering predictions'):
        row = df.iloc[i].to_dict()
        scores_i = scores[i]
        predictions[row['id']] = classes[scores_i.argmax().item()].lower()
    
    accuracy = np.mean([predictions[k] == df[df['id'] == k].iloc[0]['class'] for k in df.id.tolist()])
    print(f"Accuracy: {accuracy:.2f}")
