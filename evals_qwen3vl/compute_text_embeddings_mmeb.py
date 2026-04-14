"""
Pre-compute and save Qwen3VL-Embedding text embeddings for MMEB-V2 CLS and RET tasks.

Saved to:
  {feat_dir}/{model_name}_text_embeddings_mmebv2_text_cls.pt
  {feat_dir}/{model_name}_text_embeddings_mmebv2_text_ret.pt

Each file is a dict:  { ds_key: { text_string: tensor (D,) } }

Usage:
    python evals_qwen3vl/compute_text_embeddings_mmeb.py --task cls
    python evals_qwen3vl/compute_text_embeddings_mmeb.py --task ret
    python evals_qwen3vl/compute_text_embeddings_mmeb.py --task all
"""

import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'False'

import argparse

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import shared.utils as su
from models.qwen3vl_embedding import Qwen3VLEmbedder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT   = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
FEAT_DIR    = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2/features'
CLS_CFG     = '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
RET_CFG     = '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml'
VIDEO_ROOT  = f'{DATA_ROOT}/video-tasks/frames/data/ziyan/video_retrieval'

RET_JSON_PATHS = {
    'MSR-VTT':  ('VLM2Vec/MSR-VTT',  'test_1k', 'test'),
    'MSVD':     ('VLM2Vec/MSVD',      None,      'test'),
    'DiDeMo':   ('VLM2Vec/DiDeMo',    None,      'test'),
    'YouCook2': ('lmms-lab/YouCook2', None,      'val'),
    'VATEX':    ('VLM2Vec/VATEX',     None,      'test'),
}
RET_VIDEO_ID_EXTRACTOR = {
    'MSR-VTT':  lambda x: x['video_id'],
    'MSVD':     lambda x: x['video_id'],
    'DiDeMo':   lambda x: x['video'].split('/')[-1].split('.')[0],
    'YouCook2': lambda x: x['id'],
    'VATEX':    lambda x: x['videoID'],
}
RET_CAPTIONS_EXTRACTOR = {
    'MSR-VTT':  lambda x: [x['caption']],
    'MSVD':     lambda x: x['caption'],
    'DiDeMo':   lambda x: [x['caption']],
    'YouCook2': lambda x: [x['sentence']],
    'VATEX':    lambda x: x['enCap'],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gather_text_embeddings(model, texts, desc=''):
    """Return dict {text: normalized float32 tensor (D,)}.

    Uses model.process([{'text': text}]) which internally applies the
    default instruction "Represent the user's input." – consistent with
    how video embeddings were computed for the same model.
    """
    result = {}
    for text in su.log.tqdm_iterator(texts, desc=desc or 'Text embeddings'):
        with torch.no_grad():
            emb = model.process([{'text': text}])
            zt = emb.squeeze(0).cpu().float()
        result[text] = zt
    return result


# ---------------------------------------------------------------------------
# CLS task
# ---------------------------------------------------------------------------

def compute_cls_text_embeddings(model):
    """
    Returns dict  { ds_key: { text: tensor } }

    SmthSmthV2  – all unique neg_text options (multiple-choice pool).
    HMDB51 / UCF101 / K700 / Breakfast – all unique pos_text class labels.
    """
    meta_config = su.io.load_yml(CLS_CFG)
    text_embeds = {}

    # SmthSmthV2: multiple-choice pool
    ds_key = 'SmthSmthV2'
    su.log.print_update(f'[CLS] {ds_key}')
    d = meta_config[ds_key]
    data = pd.DataFrame(su.io.load_jsonl(f"{DATA_ROOT}/video-tasks/data/{d['json_name']}"))
    all_texts = np.unique(data.neg_text.sum()).tolist()
    print(f'  {len(all_texts)} unique texts')
    text_embeds[ds_key] = gather_text_embeddings(model, all_texts, desc=ds_key)

    # Global classification datasets
    for ds_key in ['HMDB51', 'UCF101', 'K700', 'Breakfast']:
        su.log.print_update(f'[CLS] {ds_key}')
        d = meta_config[ds_key]
        data = pd.DataFrame(su.io.load_jsonl(f"{DATA_ROOT}/video-tasks/data/{d['json_name']}"))
        all_texts = data.pos_text.unique().tolist()
        print(f'  {len(all_texts)} unique class labels')
        text_embeds[ds_key] = gather_text_embeddings(model, all_texts, desc=ds_key)
        su.log.print_update('')

    return text_embeds


# ---------------------------------------------------------------------------
# RET task
# ---------------------------------------------------------------------------

def compute_ret_text_embeddings(model):
    """
    Returns dict  { ds_key: { text: tensor } }
    """
    from datasets import load_dataset

    meta_config = su.io.load_yml(RET_CFG)
    text_embeds = {}

    for ds_key in meta_config:
        su.log.print_update(f'[RET] {ds_key}')
        repo, subset, split = RET_JSON_PATHS[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])
        df['video_id'] = df.apply(lambda x: RET_VIDEO_ID_EXTRACTOR[ds_key](x), axis=1)

        all_texts = [
            RET_CAPTIONS_EXTRACTOR[ds_key](df.iloc[i].to_dict())
            for i in range(len(df))
        ]
        all_texts = np.unique(np.concatenate(all_texts)).tolist()
        print(f'  {len(all_texts)} unique captions')
        text_embeds[ds_key] = gather_text_embeddings(model, all_texts, desc=ds_key)
        su.log.print_update('')

    return text_embeds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pre-compute Qwen3VL-Embedding text embeddings for MMEB-V2'
    )
    parser.add_argument(
        '--model_path', type=str,
        default='/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B',
    )
    parser.add_argument('--model_name', type=str,
                        default='qwen3vlembedding-base')
    parser.add_argument('--task', type=str, default='all',
                        choices=['cls', 'ret', 'all'])
    parser.add_argument('--feat_dir', type=str, default=FEAT_DIR)
    parser.add_argument('--device_map', type=str, default='cuda:0')
    args = parser.parse_args()

    os.makedirs(args.feat_dir, exist_ok=True)

    tasks = ['cls', 'ret'] if args.task == 'all' else [args.task]

    # Check which tasks already have saved embeddings
    pending = []
    for task in tasks:
        save_path = os.path.join(
            args.feat_dir,
            f'{args.model_name}_text_embeddings_mmebv2_text_{task}.pt',
        )
        if os.path.exists(save_path):
            print(f'[{task.upper()}] Already exists, skipping: {save_path}')
        else:
            pending.append(task)

    if not pending:
        print('All requested embeddings already computed. Exiting.')
        sys.exit(0)

    # Load model once, compute all pending tasks
    print(f'\nLoading Qwen3VLEmbedder from {args.model_path}')
    model = Qwen3VLEmbedder(
        model_name_or_path=args.model_path,
        torch_dtype=torch.float16,
        attn_implementation='flash_attention_2',
        device_map=args.device_map,
    )
    su.misc.num_params(model.model)

    for task in pending:
        save_path = os.path.join(
            args.feat_dir,
            f'{args.model_name}_text_embeddings_mmebv2_text_{task}.pt',
        )
        print(f'\n=== Computing text embeddings for task={task} ===')

        if task == 'cls':
            text_embeds = compute_cls_text_embeddings(model)
        else:
            text_embeds = compute_ret_text_embeddings(model)

        # Summary
        total = sum(len(v) for v in text_embeds.values())
        print(f'\nTotal unique texts embedded: {total}')
        for ds_key, embs in text_embeds.items():
            first_val = next(iter(embs.values()))
            print(f'  {ds_key}: {len(embs)} texts, dim={first_val.shape[0]}')

        torch.save(text_embeds, save_path)
        print(f'Saved → {save_path}')
