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


def load_data_video_cls(
    data_root='/scratch/shared/beegfs/piyush/datasets/MMEB-V2',
    cfg_path='/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
):
    # Load meta config
    meta_config = su.io.load_yml(cfg_path)

    # Generate dataframe of video paths
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
    return df_video


def load_data_video_ret(
    data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2',
    cfg_path = '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml',
    video_root = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames/data/ziyan/video_retrieval"
):
    # Load meta config
    meta_config = su.io.load_yml(cfg_path)

    # Load video root
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
    return df_video


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/')
    parser.add_argument('--model_name', type=str, default='tarsier2_7b')
    parser.add_argument("--task", type=str, default='cls', choices=['cls', 'ret'])
    args = parser.parse_args()

    # Load data
    if args.task == 'cls':
        df = load_data_video_cls()
    elif args.task == 'ret':
        df = load_data_video_ret()
    else:
        raise ValueError(f"Invalid task: {args.task}")
    df['video_path'] = df['video_dir'].apply(lambda x: x.replace('video-tasks/frames', 'video-tasks/videos') + '.mp4')
    df = df[df['video_path'].apply(os.path.exists)]
    print(f"Loaded {len(df)} video paths for {args.task}")


    # Load model
    model = AutoEncoder.from_pretrained(
        args.model_path,
        device_map='auto',
        attn_implementation='flash_attention_2',
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)
    
    
    # Compute video embeddings
    video_embeddings = {}
    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing video embeddings'):
        row = df.iloc[i].to_dict()
        video_path = row['video_path']
        try:
            zv = model.encode_vision(video_path).cpu().squeeze(0).float()
            video_embeddings[row['video_id']] = zv
        except:
            print(f"Error computing video embedding for {row['video_id']}")
            continue
    save_dir = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/features"
    save_name = f"{args.model_name}_video_embeddings_mmebv2_video_{args.task}.pt"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(video_embeddings, os.path.join(save_dir, save_name))
    print(f"Saved video embeddings to {os.path.join(save_dir, save_name)}")