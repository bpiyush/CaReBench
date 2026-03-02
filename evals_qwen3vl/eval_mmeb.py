import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import json

import shared.utils as su
from models.qwen3vl_embedding import Qwen3VLEmbedder


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data_video_cls(
    data_root='/scratch/shared/beegfs/piyush/datasets/MMEB-V2',
    cfg_path='/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
):
    meta_config = su.io.load_yml(cfg_path)

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
    data_root='/scratch/shared/beegfs/piyush/datasets/MMEB-V2',
    cfg_path='/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml',
    video_root="/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames/data/ziyan/video_retrieval"
):
    meta_config = su.io.load_yml(cfg_path)

    json_paths = {
        "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
        "MSVD": ("VLM2Vec/MSVD", None, "test"),
        "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
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

        df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)
        df['video_dir'] = df['video_id'].apply(lambda x: f"{video_root}/{ds_key}/frames/{x}")
        df_video['video_id'].extend(df['video_id'].tolist())
        df_video['video_dir'].extend(df['video_dir'].tolist())
        print('-' * 100)
    df_video = pd.DataFrame(df_video)
    assert len(df_video.video_id.unique()) == len(df_video)
    return df_video


# ---------------------------------------------------------------------------
# Feature computation helpers
# ---------------------------------------------------------------------------

def gather_video_embeddings(model, df):
    """Compute normalized video embeddings keyed by video_id."""
    video_embeddings = {}
    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing video embeddings'):
        row = df.iloc[i].to_dict()
        video_path = row['video_path']
        video_id = row['video_id']
        try:
            emb = model.process([{'video': video_path}])
            zv = emb.squeeze(0).cpu().float()
            video_embeddings[video_id] = zv
        except Exception as e:
            print(f"Error computing video embedding for {video_id}: {e}")
            continue
    return video_embeddings


def gather_text_embeddings(model, texts, ds_name):
    """Compute normalized text embeddings keyed by text string."""
    text_to_emb = {}
    for text in su.log.tqdm_iterator(texts, desc=f'Computing text embeddings for {ds_name}'):
        with torch.no_grad():
            emb = model.process([{'text': text}])
            zt = emb.squeeze(0).cpu().float()
        text_to_emb[text] = zt
    return text_to_emb


# ---------------------------------------------------------------------------
# Evaluation: Classification
# ---------------------------------------------------------------------------

def eval_cls(model, video_embs):
    data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
    meta_config = su.io.load_yml(
        '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
    )

    # --- SmthSmthV2 (multiple-choice format) ---
    ds_key = "SmthSmthV2"
    su.log.print_update(f"Processing {ds_key}")
    d = meta_config[ds_key]
    data = su.io.load_jsonl(f"{data_root}/video-tasks/data/{d['json_name']}")
    data = pd.DataFrame(data)
    all_texts = np.unique(data.neg_text.sum())
    text_to_emb = gather_text_embeddings(model, all_texts, ds_key)

    correct = []
    for j in su.log.tqdm_iterator(range(len(data)), desc='Gathering predictions'):
        row = data.iloc[j].to_dict()
        if row['video_id'] not in video_embs:
            continue
        texts = row['neg_text']
        zt = torch.stack([text_to_emb[t] for t in texts])
        gt_index = texts.index(row['pos_text'])
        sim = video_embs[row['video_id']] @ zt.T
        pred_index = sim.argmax().item()
        correct.append(int(gt_index == pred_index))
    accuracy = np.mean(correct)
    accuracies = {'SmthSmthV2': np.round(accuracy * 100, 2)}

    # --- Other datasets (nearest-neighbour among all class labels) ---
    for ds_key in ['HMDB51', 'UCF101', 'K700', 'Breakfast']:
        su.log.print_update(f"Processing {ds_key}")
        d = meta_config[ds_key]
        data = su.io.load_jsonl(f"{data_root}/video-tasks/data/{d['json_name']}")
        data = pd.DataFrame(data)
        print("Number of rows: ", len(data))

        data = data[data.video_id.apply(lambda x: x in set(video_embs))]
        print("Number of rows after filtering: ", len(data))

        zv = torch.stack([video_embs[c] for c in data.video_id.tolist()])
        texts_local = data.pos_text.unique()
        text_to_emb_local = gather_text_embeddings(model, texts_local, ds_key)
        zt = torch.stack([text_to_emb_local[c] for c in data.pos_text.tolist()])

        sim = zv @ zt.T
        pred_indices = sim.argmax(dim=-1)
        pred_classes = [data.pos_text.tolist()[i] for i in pred_indices]
        accuracy = np.round(
            (np.array(pred_classes) == np.array(data.pos_text)).mean() * 100, 2
        )
        accuracies[ds_key] = accuracy
        su.log.print_update(f"")

    mean_accuracy = np.mean(list(accuracies.values()))
    print(f"\n[CLS] Per-dataset accuracies: {accuracies}")
    print(f"[CLS] Mean accuracy: {mean_accuracy:.2f}")
    return accuracies


# ---------------------------------------------------------------------------
# Evaluation: Retrieval
# ---------------------------------------------------------------------------

def eval_ret(model, video_embs):
    data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
    meta_config = su.io.load_yml(
        '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml'
    )

    json_paths = {
        "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
        "MSVD": ("VLM2Vec/MSVD", None, "test"),
        "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
        "YouCook2": ("lmms-lab/YouCook2", None, "val"),
        "VATEX": ("VLM2Vec/VATEX", None, "test"),
    }
    video_id_extractor = {
        "MSR-VTT": lambda x: x['video_id'],
        "MSVD": lambda x: x['video_id'],
        "DiDeMo": lambda x: x['video'].split('/')[-1].split('.')[0],
        "YouCook2": lambda x: x["id"],
        "VATEX": lambda x: x['videoID'],
    }
    video_root = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames/data/ziyan/video_retrieval"
    captions_extractor = {
        "MSR-VTT": lambda x: [x["caption"]],
        "MSVD": lambda x: x["caption"],
        "DiDeMo": lambda x: [x["caption"]],
        "YouCook2": lambda x: [x['sentence']],
        "VATEX": lambda x: x["enCap"],
    }

    from datasets import load_dataset

    # Pre-compute text embeddings for all datasets
    text_embeds = {}
    for ds_key in meta_config:
        su.log.print_update(f"Computing text embeddings for {ds_key}")
        repo, subset, split = json_paths[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])
        video_dir = f"{video_root}/{ds_key}/frames"
        video_ids = os.listdir(video_dir)
        assert len(video_ids) == len(df)
        df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)

        all_texts = [
            captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))
        ]
        all_texts = np.unique(np.concatenate(all_texts))
        print("Total number of unique captions: ", len(all_texts))
        text_embeds[ds_key] = gather_text_embeddings(model, all_texts, ds_key)
        su.log.print_update(f"")

    # Evaluate retrieval per dataset
    ret_accs = {}
    for ds_key in meta_config:
        su.log.print_update(f"Evaluating {ds_key}")
        repo, subset, split = json_paths[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])
        video_dir = f"{video_root}/{ds_key}/frames"
        video_ids = os.listdir(video_dir)
        assert len(video_ids) == len(df)
        df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)

        # Filter to videos with computed embeddings
        df = df[df.video_id.apply(lambda x: x in set(video_embs))]
        print(f"{ds_key}: {len(df)} videos with embeddings")

        all_texts = [
            captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))
        ]
        all_texts = np.unique(np.concatenate(all_texts))
        text_emb = text_embeds[ds_key]

        zv = torch.stack([video_embs[c] for c in df.video_id.tolist()])
        zt = torch.stack([text_emb[t] for t in all_texts])

        sim = zv @ zt.T
        pred_indices = sim.argmax(dim=-1)
        pred_captions = np.array([all_texts[i] for i in pred_indices])
        actu_captions = [
            captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))
        ]
        is_correct = [int(x in y) for x, y in zip(pred_captions, actu_captions)]
        accuracy = np.round(np.mean(is_correct) * 100., 2).item()
        ret_accs[ds_key] = accuracy
        su.log.print_update(f"")

    mean_acc = np.round(np.mean(list(ret_accs.values())), 2)
    print(f"\n[RET] Per-dataset accuracies: {ret_accs}")
    print(f"[RET] Mean accuracy: {mean_acc:.2f}")
    return ret_accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', type=str,
        default='/work/piyush/experiments/CaRe/Qwen3-VL-Embedding-8B/'
                'final-10112025/nli_9000+ego_1000+subj_replaced-seed_42',
    )
    parser.add_argument('--model_name', type=str, default='qwen3vlembedding-finetuned')
    parser.add_argument("--task", type=str, default='cls', choices=['cls', 'ret'])
    args = parser.parse_args()

    # Check for cached video embeddings
    save_dir = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/features"
    save_name = f"{args.model_name}_video_embeddings_mmebv2_video_{args.task}.pt"
    save_path = os.path.join(save_dir, save_name)

    if os.path.exists(save_path):
        print(f"Loading cached video embeddings from {save_path}")
        video_embs = torch.load(save_path)
        print(f"Loaded embeddings for {len(video_embs)} videos")
    else:
        # Load data
        if args.task == 'cls':
            df = load_data_video_cls()
        elif args.task == 'ret':
            df = load_data_video_ret()
        else:
            raise ValueError(f"Invalid task: {args.task}")

        df['video_path'] = df['video_dir'].apply(
            lambda x: x.replace('video-tasks/frames', 'video-tasks/videos') + '.mp4'
        )
        df = df[df['video_path'].apply(os.path.exists)]
        print(f"Loaded {len(df)} video paths for task={args.task}")

        # Load model for video embedding computation
        model = Qwen3VLEmbedder(
            model_name_or_path=args.model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )
        su.misc.num_params(model.model)

        print("Computing video embeddings...")
        video_embs = gather_video_embeddings(model, df)
        print(f"Computed embeddings for {len(video_embs)} videos")

        os.makedirs(save_dir, exist_ok=True)
        torch.save(video_embs, save_path)
        print(f"Saved video embeddings to {save_path}")

    # Load model for text embeddings (needed for evaluation regardless)
    if 'model' not in locals():
        model = Qwen3VLEmbedder(
            model_name_or_path=args.model_path,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
        )
        su.misc.num_params(model.model)

    # Evaluate
    if args.task == 'cls':
        accuracies = eval_cls(model, video_embs)
    elif args.task == 'ret':
        ret_accs = eval_ret(model, video_embs)
