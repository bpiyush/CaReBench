import os
import torch
import argparse

import shared.utils as su
from models.qwen3vl_embedding import Qwen3VLEmbedder
from tasks.eval_chiral_retrieval import load_data
from utils.chiral_retrieval_metrics import (
    compute_metrics,
    print_metrics_as_latex_row,
)
import pandas as pd
import json


def gather_text_features(model, texts):
    """Compute text embeddings for all unique text IDs."""
    texts_feat = {}
    for text in enumerate(
        su.log.tqdm_iterator(texts, desc='Computing text features')
    ):
        emb = model.process([{'text': text}])
        zt = emb.squeeze(0).cpu().float()
        texts_feat[text] = zt
    return texts_feat


def gather_video_features(
    df,
    model,
    # fps,
    # max_frames,
):
    """Compute video embeddings for all unique video IDs."""
    video_ids = df.video_id.unique()
    video_feat = {}
    for j, video_id in enumerate(
        su.log.tqdm_iterator(video_ids, desc='Computing video features')
    ):
        video_path = df[df.video_id == video_id].video_path.unique()[0]

        try:
            emb = model.process([{
                'video': video_path,
                # 'fps': fps,
                # 'max_frames': max_frames,
                'nframes': 16,
            }])
            zv = emb.squeeze(0).cpu().float()
            video_feat[video_id] = zv
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
    return video_feat


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


def configure_save_path(model_name, task):
    save_dir = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/features"
    save_name = f"{model_name}_video_embeddings_mmebv2_video_{task}.pt"
    os.makedirs(save_dir, exist_ok=True)
    return os.path.join(save_dir, save_name)


if __name__ == "__main__":
    # model_path = "/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B"
    model_paths = [
        "/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B",
        "/work/piyush/experiments/CaRe/Qwen3-VL-Embedding-8B/final-10112025/nli_9000+ego_1000+subj_replaced-seed_42/",
    ]
    model_names = [
        "qwen3vlembedding-base",
        # "qwen3vlembedding-finetuned",
    ]
    
    for model_path, model_name in zip(model_paths, model_names):
        
        print(f"Loading model from {model_path}")

        model = Qwen3VLEmbedder(
            model_name_or_path=model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )
        su.misc.num_params(model.model)
        
        
        #########################################################
        # Compute video embeddings for CLS task
        #########################################################
        df = load_data_video_cls()
        df['video_path'] = df['video_dir'].apply(lambda x: x.replace('video-tasks/frames', 'video-tasks/videos') + '.mp4')
        df = df[df['video_path'].apply(os.path.exists)]
        
        save_path_cls = configure_save_path(model_name, "cls")
        if os.path.exists(save_path_cls):
            print(f"Loading cached video embeddings from {save_path_cls}")
            video_embeddings = torch.load(save_path_cls)
            print(f"Loaded embeddings for {len(video_embeddings)} videos")
            # video_embeds_new  = {}
            # for k in video_embeddings.keys():
            #     if isinstance(k, str):
            #         video_embeds_new[os.path.basename(str(k)).split(".mp4")[0]] = video_embeddings[k]
            #     else:
            #         video_embeds_new[k] = video_embeddings[k]
            # video_embeddings = {os.path.basename(str(k)).split(".mp4")[0]: v for k, v in video_embeddings.items()}

            # Remove these from the df since we already have the embeddings
            df = df[~df['video_id'].isin(video_embeddings.keys())]
            print(f"Remaining {len(df)} videos to compute embeddings for")
        
        # Compute embeddings for remaining videos
        video_embeddings.update(gather_video_features(df, model))
        
        # Save embeddings
        torch.save(video_embeddings, save_path_cls)
        print(f"Saved ({len(video_embeddings)}) CLS video embeddings to {save_path_cls}")


        #########################################################
        # Compute video embeddings for RET task
        #########################################################
        df = load_data_video_ret()
        df['video_path'] = df['video_dir'].apply(lambda x: x.replace('video-tasks/frames', 'video-tasks/videos') + '.mp4')
        df = df[df['video_path'].apply(os.path.exists)]

        save_path_ret = configure_save_path(model_name, "ret")
        if os.path.exists(save_path_ret):
            print(f"Loading cached video embeddings from {save_path_ret}")
            video_embeddings = torch.load(save_path_ret)
            print(f"Loaded embeddings for {len(video_embeddings)} videos")

            # Remove these from the df_ret since we already have the embeddings
            df = df[~df['video_id'].isin(video_embeddings.keys())]
            print(f"Remaining {len(df)} videos to compute embeddings for")
        
        # Compute embeddings for remaining videos
        video_embeddings.update(gather_video_features(df, model))

        # Save embeddings
        torch.save(video_embeddings, save_path_ret)
        print(f"Saved ({len(video_embeddings)}) RET video embeddings to {save_path_ret}")