import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"
from glob import glob

import torch
import pandas as pd
import numpy as np
import json

import shared.utils as su
from models.modeling_encoders import AutoEncoder


VIDEO_DIRS = {
    "VATEX": f"/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/videos/data/ziyan/video_retrieval/VATEX/frames",
    "MSR-VTT": f"/scratch/shared/beegfs/piyush/datasets/MSRVTT/videos/all",
    "ActivityNet": "/scratch/shared/beegfs/piyush/datasets/Tarsier2/ActivityNet/videos",
}

def embed_adverb_action(encoder, adverb, action, verbose=False):
    prompt = f"The action {action} is performed {adverb}."
    with torch.no_grad():
        zt = encoder.encode_text(prompt).cpu().squeeze(0).float().cpu()
        zt = torch.nn.functional.normalize(zt, dim=-1)
    return zt


def embed_video_action(model, video_path, action, verbose=False):

    PROMPT = "Video: <video>\n"\
             "Action: This video shows the action: <sent>\n"\
             "Look at the video carefully.\n"\
             "Summarize the action in the video in one word:"
    PROMPT = f"USER: {PROMPT} ASSISTANT: "
    input_prompt = PROMPT.replace("<sent>", action)
    zv = model.encode_vision_with_text(video_path, input_prompt)
    zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float().squeeze(0)
    return zv


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/')
    parser.add_argument('--model_name', type=str, default='tarsier2_7b')
    parser.add_argument("--ds", type=str, default='VATEX', choices=['VATEX', 'MSR-VTT', 'ActivityNet'])
    args = parser.parse_args()

    data_dir = "/scratch/shared/beegfs/piyush/datasets/PseudoAdverbs"
    df_anno = pd.read_csv(f"{data_dir}/datasets/{args.ds}_Adverbs/annotations.csv")
    df_advb = pd.read_csv(f"{data_dir}/datasets/{args.ds}_Adverbs/adverbs.csv")
    video_dir = VIDEO_DIRS[args.ds]
    df_anno['clip_path'] = df_anno.clip_id.apply(lambda x: f"{video_dir}/{x}.mp4")
    print("Number of rows in CSV: ", len(df_anno))
    df_anno = df_anno[df_anno['clip_path'].apply(os.path.exists)]
    print("Number of rows with clip directory available: ", len(df_anno))
    adverb_to_antonym = dict(df_advb.values)


    # Load model
    model = AutoEncoder.from_pretrained(
        args.model_path,
        device_map='auto',
        attn_implementation='flash_attention_2',
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)
    
    # Test on sample
    row = df_anno.iloc[0].to_dict()
    zv = embed_video_action(model, row['clip_path'], row['clustered_action'])
    zt = embed_adverb_action(model, row['clustered_adverb'], row['clustered_action'])
    print(zv.shape, zt.shape)

    
    # Compute video embeddings
    video_embeds = {}
    for i in su.log.tqdm_iterator(range(len(df_anno)), desc='Computing video embeddings'):
        row = df_anno.iloc[i].to_dict()
        
        try:
            zv = embed_video_action(model, row['clip_path'], row['clustered_action'])
            video_embeds[row['clip_id']] = zv
        except:
            print(f"Error computing video embedding for {row['clip_id']}")
            continue

    
    # Compute text embeddings
    text_embeds = {}
    for i in su.log.tqdm_iterator(
        range(len(df_anno)), desc='Computing features for text'
    ):
        row = df_anno.iloc[i].to_dict()
        key = f"{row['clustered_action']}/{row['clustered_adverb']}"
        if key in text_embeds:
            continue
        zv = embed_adverb_action(model, row['clustered_adverb'], row['clustered_action'])
        zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float()
        text_embeds[key] = zv

        antonym = adverb_to_antonym[row['clustered_adverb']]
        key = f"{row['clustered_action']}/{antonym}"
        if key in text_embeds:
            continue
        zv = embed_adverb_action(model, antonym, row['clustered_action'])
        zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float()
        text_embeds[key] = zv

    import ipdb; ipdb.set_trace()

    correct = []
    for i in range(len(df_anno)):
        row = df_anno.iloc[i].to_dict()
        clip_id = row['clip_id']
        action = row['clustered_action']
        adverb = row['clustered_adverb']
        try:
            zt_adverb = text_embeds[f"{action}/{adverb}"]
            zt_antonm = text_embeds[f"{action}/{adverb_to_antonym[adverb]}"]
            zv = video_embeds[clip_id]
            c = (zv @ zt_adverb) > (zv @ zt_antonm)
            correct.append(int(c))
        except:
            print(i)
            continue
    accu = np.round(100. * np.mean(correct), 3)
    print("Accuracy: ", accu)