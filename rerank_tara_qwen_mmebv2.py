"""
Re-ranking on MMEB-V2: Qwen3-VL-Embedding retrieves top-K, TARA re-ranks within top-K.

Strategy:
  1. Compute full similarity with Qwen for all gallery texts.
  2. Keep top-K candidates (K = 10% of gallery, min 2).
  3. Re-rank those K candidates using TARA similarity.
  4. Final prediction = top-1 after TARA re-ranking.

Usage:
  python rerank_tara_qwen_mmebv2.py
  python rerank_tara_qwen_mmebv2.py --tara_model_name tarsier2-tara-cia10k-covr10k
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import shared.utils as su  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEAT_ROOT  = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2/features'
DATA_ROOT  = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
CLS_CFG    = '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
RET_CFG    = '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml'

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

K_FRACTION = 0.10
K_MIN = 2
K_FIXED = None   # set via --k_fixed; overrides fraction-based K when not None


def get_k(gallery_size: int) -> int:
    if K_FIXED is not None:
        return min(K_FIXED, gallery_size)
    return max(K_MIN, int(K_FRACTION * gallery_size))

QWEN_CLS = {'SmthSmthV2': 76.9, 'HMDB51': 77.5, 'UCF101': 94.6,
            'K700': 68.4, 'Breakfast': 67.36}
QWEN_RET = {'MSR-VTT': 53.8, 'MSVD': 87.16, 'DiDeMo': 56.18,
            'YouCook2': 32.9, 'VATEX': 64.78}
TARA_CLS = {'SmthSmthV2': 76.4, 'HMDB51': 69.0, 'UCF101': 80.3,
            'K700': 59.4, 'Breakfast': 45.6}
TARA_RET = {'MSR-VTT': 40.7, 'MSVD': 82.2, 'DiDeMo': 36.8,
            'YouCook2': 16.7, 'VATEX': 53.2}


def topk_rerank(sim_qwen: torch.Tensor, sim_tara: torch.Tensor, k: int) -> torch.Tensor:
    """For each query row: retrieve top-k with Qwen, re-rank with TARA.
    Returns predicted text index per query, shape (N_queries,)."""
    k = max(K_MIN, min(k, sim_qwen.shape[1]))
    topk_idx = sim_qwen.topk(k, dim=-1).indices       # (N, k)
    tara_topk = sim_tara.gather(1, topk_idx)           # (N, k)
    best_within = tara_topk.argmax(dim=-1)             # (N,)
    return topk_idx[torch.arange(sim_qwen.shape[0]), best_within]


def load_embeddings(path: str, label: str) -> dict:
    assert os.path.exists(path), f'{label} file not found: {path}'
    data = torch.load(path, weights_only=False)
    print(f'  {label}: {len(data)} entries  [{os.path.basename(path)}]')
    return data


# ---------------------------------------------------------------------------
# CLS
# ---------------------------------------------------------------------------

def eval_cls_rerank(vid_tara, vid_qwen, txt_tara, txt_qwen):
    meta_config = su.io.load_yml(CLS_CFG)
    valid_vid_ids = set(vid_tara) & set(vid_qwen)
    accuracies = {}

    # SmthSmthV2: per-sample multiple-choice candidate set
    ds_key = 'SmthSmthV2'
    su.log.print_update(f'[CLS] {ds_key}')
    d = meta_config[ds_key]
    data = pd.DataFrame(su.io.load_jsonl(
        f"{DATA_ROOT}/video-tasks/data/{d['json_name']}"))
    t_tara = txt_tara[ds_key]
    t_qwen = txt_qwen[ds_key]

    correct_qwen, correct_rerank = [], []
    skipped = 0
    for j in su.log.tqdm_iterator(range(len(data)), desc='Predictions'):
        row = data.iloc[j].to_dict()
        vid_id = row['video_id']
        if vid_id not in valid_vid_ids:
            skipped += 1
            continue
        texts = row['neg_text']
        gt_index = texts.index(row['pos_text'])
        n = len(texts)
        k = get_k(n)

        zt_t = torch.stack([t_tara[t] for t in texts])
        zt_q = torch.stack([t_qwen[t] for t in texts])
        sim_q = vid_qwen[vid_id] @ zt_q.T
        sim_t = vid_tara[vid_id] @ zt_t.T

        correct_qwen.append(int(sim_q.argmax().item() == gt_index))

        topk = sim_q.topk(k).indices
        pred = topk[sim_t[topk].argmax()].item()
        correct_rerank.append(int(pred == gt_index))

    k_example = get_k(len(texts))
    acc_q = float(np.round(np.mean(correct_qwen) * 100, 2))
    acc_r = float(np.round(np.mean(correct_rerank) * 100, 2))
    print(f'  Qwen-only: {acc_q:.2f}%  |  Re-ranked (K~{k_example}): {acc_r:.2f}%')
    accuracies[ds_key] = {'qwen': acc_q, 'rerank': acc_r}

    # HMDB51 / UCF101 / K700 / Breakfast
    for ds_key in ['HMDB51', 'UCF101', 'K700', 'Breakfast']:
        su.log.print_update(f'[CLS] {ds_key}')
        d = meta_config[ds_key]
        data = pd.DataFrame(su.io.load_jsonl(
            f"{DATA_ROOT}/video-tasks/data/{d['json_name']}"))
        data = data[data.video_id.apply(lambda x: x in valid_vid_ids)]

        t_tara_ds = txt_tara[ds_key]
        t_qwen_ds = txt_qwen[ds_key]
        all_classes = list(t_qwen_ds.keys())
        N_gallery = len(all_classes)
        k = get_k(N_gallery)
        print(f'  gallery={N_gallery} classes, K={k}')

        zv_t = torch.stack([vid_tara[v] for v in data.video_id])
        zv_q = torch.stack([vid_qwen[v] for v in data.video_id])
        zt_t = torch.stack([t_tara_ds[c] for c in all_classes])
        zt_q = torch.stack([t_qwen_ds[c] for c in all_classes])

        sim_q = zv_q @ zt_q.T
        sim_t = zv_t @ zt_t.T
        gt = np.array(data.pos_text.tolist())

        pred_q = [all_classes[i] for i in sim_q.argmax(dim=-1).tolist()]
        acc_q = float(np.round((np.array(pred_q) == gt).mean() * 100, 2))

        pred_r = [all_classes[i] for i in topk_rerank(sim_q, sim_t, k).tolist()]
        acc_r = float(np.round((np.array(pred_r) == gt).mean() * 100, 2))

        print(f'  Qwen-only: {acc_q:.2f}%  |  Re-ranked (K={k}): {acc_r:.2f}%')
        accuracies[ds_key] = {'qwen': acc_q, 'rerank': acc_r}
        su.log.print_update('')

    return accuracies


# ---------------------------------------------------------------------------
# RET
# ---------------------------------------------------------------------------

def eval_ret_rerank(vid_tara, vid_qwen, txt_tara, txt_qwen):
    from datasets import load_dataset
    meta_config = su.io.load_yml(RET_CFG)
    valid_vid_ids = set(vid_tara) & set(vid_qwen)
    ret_accs = {}

    for ds_key in meta_config:
        su.log.print_update(f'[RET] {ds_key}')
        repo, subset, split = RET_JSON_PATHS[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])
        df['video_id'] = df.apply(
            lambda x: RET_VIDEO_ID_EXTRACTOR[ds_key](x), axis=1)
        df = df[df.video_id.apply(lambda x: x in valid_vid_ids)]

        all_texts = np.unique(np.concatenate([
            RET_CAPTIONS_EXTRACTOR[ds_key](df.iloc[i].to_dict())
            for i in range(len(df))
        ]))
        N_gallery = len(all_texts)
        k = get_k(N_gallery)
        print(f'  {ds_key}: {len(df)} videos, gallery={N_gallery} texts, K={k}')

        t_tara_ds = txt_tara[ds_key]
        t_qwen_ds = txt_qwen[ds_key]

        zv_t = torch.stack([vid_tara[v] for v in df.video_id])
        zv_q = torch.stack([vid_qwen[v] for v in df.video_id])
        zt_t = torch.stack([t_tara_ds[t] for t in all_texts])
        zt_q = torch.stack([t_qwen_ds[t] for t in all_texts])

        sim_q = zv_q @ zt_q.T
        sim_t = zv_t @ zt_t.T

        actu = [RET_CAPTIONS_EXTRACTOR[ds_key](df.iloc[i].to_dict())
                for i in range(len(df))]

        pred_q = sim_q.argmax(dim=-1).tolist()
        acc_q = float(np.round(
            np.mean([int(all_texts[p] in a) for p, a in zip(pred_q, actu)]) * 100, 2))

        pred_r = topk_rerank(sim_q, sim_t, k).tolist()
        acc_r = float(np.round(
            np.mean([int(all_texts[p] in a) for p, a in zip(pred_r, actu)]) * 100, 2))

        print(f'  Qwen-only: {acc_q:.2f}%  |  Re-ranked (K={k}): {acc_r:.2f}%')
        ret_accs[ds_key] = {'qwen': acc_q, 'rerank': acc_r}
        su.log.print_update('')

    return ret_accs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tara_model_name', default='tarsier2-tara-cia10k-covr10k')
    parser.add_argument('--qwen_model_name',  default='qwen3vlembedding-base')
    parser.add_argument('--task', default='all', choices=['cls', 'ret', 'all'])
    parser.add_argument('--feat_root', default=FEAT_ROOT)
    parser.add_argument('--k_fixed', type=int, default=None,
                        help='Use a fixed K for all datasets instead of 10%% of gallery.')
    args = parser.parse_args()

    global K_FIXED
    if args.k_fixed is not None:
        K_FIXED = args.k_fixed
        print(f'Using fixed K={K_FIXED} for all datasets.')

    tasks = ['cls', 'ret'] if args.task == 'all' else [args.task]

    def feat_path(model, modality, task):
        return os.path.join(
            args.feat_root,
            f'{model}_{modality}_embeddings_mmebv2_{modality}_{task}.pt')

    print('\n=== Loading embeddings ===')
    vid_embs, txt_embs = {}, {}
    for task in tasks:
        print(f'\n-- Video [{task.upper()}] --')
        v_tara = load_embeddings(feat_path(args.tara_model_name, 'video', task), 'TARA')
        v_qwen = load_embeddings(feat_path(args.qwen_model_name,  'video', task), 'Qwen')
        print(f'  Intersection: {len(set(v_tara) & set(v_qwen))} videos')
        vid_embs[task] = (v_tara, v_qwen)

        print(f'\n-- Text [{task.upper()}] --')
        t_tara = load_embeddings(feat_path(args.tara_model_name, 'text', task), 'TARA-text')
        t_qwen = load_embeddings(feat_path(args.qwen_model_name,  'text', task), 'Qwen-text')
        txt_embs[task] = (t_tara, t_qwen)

    results = {}
    if 'cls' in tasks:
        print('\n=== CLS Re-ranking ===')
        v_tara, v_qwen = vid_embs['cls']
        t_tara, t_qwen = txt_embs['cls']
        results['cls'] = eval_cls_rerank(v_tara, v_qwen, t_tara, t_qwen)

    if 'ret' in tasks:
        print('\n=== RET Re-ranking ===')
        v_tara, v_qwen = vid_embs['ret']
        t_tara, t_qwen = txt_embs['ret']
        results['ret'] = eval_ret_rerank(v_tara, v_qwen, t_tara, t_qwen)

    # Summary table
    print('\n' + '=' * 72)
    print(f'  {"Dataset":<16}  {"TARA":>7}  {"Qwen":>7}  {"Re-rank":>9}  {"vs Qwen":>8}')
    print('=' * 72)

    if 'cls' in results:
        rr_vals, q_vals = [], []
        for ds, v in results['cls'].items():
            delta = v['rerank'] - QWEN_CLS[ds]
            mark = ' ▲' if delta > 0 else (' ▼' if delta < 0 else '  ')
            print(f'  CLS {ds:<12}  {TARA_CLS[ds]:>7.2f}  {QWEN_CLS[ds]:>7.2f}'
                  f'  {v["rerank"]:>9.2f}  {delta:>+7.2f}{mark}')
            rr_vals.append(v['rerank'])
            q_vals.append(QWEN_CLS[ds])
        m_rr = float(np.round(np.mean(rr_vals), 2))
        m_q  = float(np.round(np.mean(q_vals), 2))
        m_t  = float(np.round(np.mean(list(TARA_CLS.values())), 2))
        delta = m_rr - m_q
        mark = ' ▲' if delta > 0 else (' ▼' if delta < 0 else '  ')
        print(f'  {"CLS MEAN":<16}  {m_t:>7.2f}  {m_q:>7.2f}'
              f'  {m_rr:>9.2f}  {delta:>+7.2f}{mark}')

    print('-' * 72)

    if 'ret' in results:
        rr_vals, q_vals = [], []
        for ds, v in results['ret'].items():
            delta = v['rerank'] - QWEN_RET[ds]
            mark = ' ▲' if delta > 0 else (' ▼' if delta < 0 else '  ')
            print(f'  RET {ds:<12}  {TARA_RET[ds]:>7.2f}  {QWEN_RET[ds]:>7.2f}'
                  f'  {v["rerank"]:>9.2f}  {delta:>+7.2f}{mark}')
            rr_vals.append(v['rerank'])
            q_vals.append(QWEN_RET[ds])
        m_rr = float(np.round(np.mean(rr_vals), 2))
        m_q  = float(np.round(np.mean(q_vals), 2))
        m_t  = float(np.round(np.mean(list(TARA_RET.values())), 2))
        delta = m_rr - m_q
        mark = ' ▲' if delta > 0 else (' ▼' if delta < 0 else '  ')
        print(f'  {"RET MEAN":<16}  {m_t:>7.2f}  {m_q:>7.2f}'
              f'  {m_rr:>9.2f}  {delta:>+7.2f}{mark}')

    print('=' * 72)


if __name__ == '__main__':
    main()
