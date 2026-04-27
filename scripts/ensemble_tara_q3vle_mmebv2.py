"""
Ensemble TARA + Qwen3VL-Embedding on MMEB-V2 CLS and RET tasks.

  s(query, candidate) = alpha * s_TARA(q, c) + (1 - alpha) * s_Qwen(q, c)

Pre-requisites (run once to produce the .pt files):
  python evals_tarsier2/compute_text_embeddings_mmeb.py --task all
  python evals_qwen3vl/compute_text_embeddings_mmeb.py  --task all

Usage:
  python scripts/ensemble_tara_q3vle_mmebv2.py --task all --alpha 0.5
  python scripts/ensemble_tara_q3vle_mmebv2.py --task cls
  python scripts/ensemble_tara_q3vle_mmebv2.py --task ret --alpha 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import shared.utils as su  # noqa: E402

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------

FEAT_ROOT  = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2/features'
DATA_ROOT  = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
CLS_CFG    = '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
RET_CFG    = '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml'
VIDEO_ROOT = f'{DATA_ROOT}/video-tasks/frames/data/ziyan/video_retrieval'

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
# Embedding loaders
# ---------------------------------------------------------------------------

def load_embeddings(path: str, label: str) -> dict:
    assert os.path.exists(path), f'{label} file not found: {path}'
    data = torch.load(path, weights_only=False)
    print(f'  {label}: {len(data)} entries loaded from {os.path.basename(path)}')
    return data


def verify_key_overlap(embs_a: dict, embs_b: dict, label_a: str, label_b: str) -> set:
    keys_a, keys_b = set(embs_a), set(embs_b)
    common = keys_a & keys_b
    only_a = keys_a - keys_b
    only_b = keys_b - keys_a
    print(f'  {label_a}: {len(keys_a)}, {label_b}: {len(keys_b)}, '
          f'intersection: {len(common)}, '
          f'only-{label_a}: {len(only_a)}, only-{label_b}: {len(only_b)}')
    return common


# ---------------------------------------------------------------------------
# CLS evaluation
# ---------------------------------------------------------------------------

def eval_cls_ensemble(
    vid_tara: dict, vid_qwen: dict,
    txt_tara: dict, txt_qwen: dict,
    alpha: float,
) -> dict:
    """
    vid_tara/qwen : { video_id -> tensor (D,) }
    txt_tara/qwen : { ds_key  -> { text -> tensor (D,) } }
    """
    meta_config = su.io.load_yml(CLS_CFG)
    valid_vid_ids = set(vid_tara) & set(vid_qwen)
    accuracies = {}

    # --- SmthSmthV2: per-sample multiple-choice ---
    ds_key = 'SmthSmthV2'
    su.log.print_update(f'[CLS] {ds_key}')
    d = meta_config[ds_key]
    data = pd.DataFrame(su.io.load_jsonl(f"{DATA_ROOT}/video-tasks/data/{d['json_name']}"))
    t_tara = txt_tara[ds_key]
    t_qwen = txt_qwen[ds_key]

    correct = []
    skipped = 0
    for j in su.log.tqdm_iterator(range(len(data)), desc='Predictions'):
        row = data.iloc[j].to_dict()
        vid_id = row['video_id']
        if vid_id not in valid_vid_ids:
            skipped += 1
            continue
        texts = row['neg_text']
        gt_index = texts.index(row['pos_text'])

        zt_t = torch.stack([t_tara[t] for t in texts])
        zt_q = torch.stack([t_qwen[t] for t in texts])
        sim = alpha * (vid_tara[vid_id] @ zt_t.T) + (1 - alpha) * (vid_qwen[vid_id] @ zt_q.T)
        correct.append(int(sim.argmax().item() == gt_index))

    print(f'  evaluated {len(correct)} samples, skipped {skipped}')
    accuracies[ds_key] = float(np.round(np.mean(correct) * 100, 2))

    # --- Global classification: nearest-class-label ---
    for ds_key in ['HMDB51', 'UCF101', 'K700', 'Breakfast']:
        su.log.print_update(f'[CLS] {ds_key}')
        d = meta_config[ds_key]
        data = pd.DataFrame(su.io.load_jsonl(f"{DATA_ROOT}/video-tasks/data/{d['json_name']}"))
        print(f'  total rows: {len(data)}')

        data = data[data.video_id.apply(lambda x: x in valid_vid_ids)]
        print(f'  rows after intersection filter: {len(data)}')

        t_tara = txt_tara[ds_key]
        t_qwen = txt_qwen[ds_key]

        zv_t = torch.stack([vid_tara[v] for v in data.video_id])
        zv_q = torch.stack([vid_qwen[v] for v in data.video_id])

        # One text embedding per sample row (classes may repeat)
        zt_t = torch.stack([t_tara[c] for c in data.pos_text])
        zt_q = torch.stack([t_qwen[c] for c in data.pos_text])

        sim = alpha * (zv_t @ zt_t.T) + (1 - alpha) * (zv_q @ zt_q.T)
        pred_indices = sim.argmax(dim=-1)
        pred_classes = [data.pos_text.tolist()[i] for i in pred_indices]
        accuracy = float(np.round(
            (np.array(pred_classes) == np.array(data.pos_text.tolist())).mean() * 100, 2
        ))
        accuracies[ds_key] = accuracy
        su.log.print_update('')

    mean_acc = float(np.round(np.mean(list(accuracies.values())), 2))
    print(f'\n[CLS] Per-dataset: {accuracies}')
    print(f'[CLS] Mean accuracy: {mean_acc:.2f}')
    return {'per_dataset': accuracies, 'mean': mean_acc}


# ---------------------------------------------------------------------------
# RET evaluation
# ---------------------------------------------------------------------------

def eval_ret_ensemble(
    vid_tara: dict, vid_qwen: dict,
    txt_tara: dict, txt_qwen: dict,
    alpha: float,
) -> dict:
    from datasets import load_dataset

    meta_config = su.io.load_yml(RET_CFG)
    valid_vid_ids = set(vid_tara) & set(vid_qwen)
    ret_accs = {}

    for ds_key in meta_config:
        su.log.print_update(f'[RET] {ds_key}')
        repo, subset, split = RET_JSON_PATHS[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])
        df['video_id'] = df.apply(lambda x: RET_VIDEO_ID_EXTRACTOR[ds_key](x), axis=1)

        # Filter to videos present in both models
        df = df[df.video_id.apply(lambda x: x in valid_vid_ids)]
        print(f'  {ds_key}: {len(df)} videos after intersection filter')

        # Text gallery from the filtered set
        all_texts = np.unique(np.concatenate([
            RET_CAPTIONS_EXTRACTOR[ds_key](df.iloc[i].to_dict())
            for i in range(len(df))
        ]))
        print(f'  {len(all_texts)} unique captions in gallery')

        t_tara = txt_tara[ds_key]
        t_qwen = txt_qwen[ds_key]

        zv_t = torch.stack([vid_tara[v] for v in df.video_id])
        zv_q = torch.stack([vid_qwen[v] for v in df.video_id])
        zt_t = torch.stack([t_tara[t] for t in all_texts])
        zt_q = torch.stack([t_qwen[t] for t in all_texts])

        sim = alpha * (zv_t @ zt_t.T) + (1 - alpha) * (zv_q @ zt_q.T)
        pred_indices = sim.argmax(dim=-1)
        pred_captions = np.array([all_texts[i] for i in pred_indices])
        actu_captions = [
            RET_CAPTIONS_EXTRACTOR[ds_key](df.iloc[i].to_dict())
            for i in range(len(df))
        ]
        is_correct = [int(x in y) for x, y in zip(pred_captions, actu_captions)]
        accuracy = float(np.round(np.mean(is_correct) * 100., 2))
        ret_accs[ds_key] = accuracy
        su.log.print_update('')

    mean_acc = float(np.round(np.mean(list(ret_accs.values())), 2))
    print(f'\n[RET] Per-dataset: {ret_accs}')
    print(f'[RET] Mean accuracy: {mean_acc:.2f}')
    return {'per_dataset': ret_accs, 'mean': mean_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Ensemble TARA + Qwen3VL-Embedding on MMEB-V2'
    )
    parser.add_argument('--tara_model_name', type=str,
                        default='tarsier2-tara-cia10k-covr10k')
    parser.add_argument('--qwen_model_name', type=str,
                        default='qwen3vlembedding-base')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight on TARA similarity; Qwen gets (1 - alpha).')
    parser.add_argument('--task', type=str, default='all',
                        choices=['cls', 'ret', 'all'])
    parser.add_argument('--feat_root', type=str, default=FEAT_ROOT)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save metrics JSON (default: ./_ensemble_metrics).')
    args = parser.parse_args()

    assert 0.0 <= args.alpha <= 1.0, 'alpha must be in [0, 1]'
    tasks = ['cls', 'ret'] if args.task == 'all' else [args.task]

    # -----------------------------------------------------------------------
    # Load embeddings
    # -----------------------------------------------------------------------
    print('\n=== Loading embeddings ===')

    def feat_path(model_name, modality, task):
        return os.path.join(
            args.feat_root,
            f'{model_name}_{modality}_embeddings_mmebv2_{modality}_{task}.pt',
        )

    # Video embeddings (one file per task)
    vid_embs = {}
    for task in tasks:
        print(f'\n-- Video embeddings [{task.upper()}] --')
        v_tara = load_embeddings(feat_path(args.tara_model_name, 'video', task), 'TARA')
        v_qwen = load_embeddings(feat_path(args.qwen_model_name, 'video', task), 'Qwen')
        
        # Normalize the video embeddings
        v_tara = {k: torch.nn.functional.normalize(v, dim=-1) for k, v in v_tara.items()}
        v_qwen = {k: torch.nn.functional.normalize(v, dim=-1) for k, v in v_qwen.items()}

        print('  Key overlap:')
        verify_key_overlap(v_tara, v_qwen, 'TARA', 'Qwen')
        vid_embs[task] = (v_tara, v_qwen)

    # Text embeddings (one file per task)
    txt_embs = {}
    for task in tasks:
        print(f'\n-- Text embeddings [{task.upper()}] --')
        t_tara = load_embeddings(feat_path(args.tara_model_name, 'text', task), 'TARA-text')
        t_qwen = load_embeddings(feat_path(args.qwen_model_name, 'text', task), 'Qwen-text')

        # t_* are dicts { ds_key -> { text -> tensor } }; report per-dataset counts
        for ds_key in t_tara:
            n_t, n_q = len(t_tara[ds_key]), len(t_qwen.get(ds_key, {}))
            print(f'  {ds_key}: TARA {n_t} texts, Qwen {n_q} texts')

            t_tara[ds_key] = {k: torch.nn.functional.normalize(v, dim=-1) for k, v in t_tara[ds_key].items()}
            t_qwen[ds_key] = {k: torch.nn.functional.normalize(v, dim=-1) for k, v in t_qwen[ds_key].items()}
        txt_embs[task] = (t_tara, t_qwen)

    # -----------------------------------------------------------------------
    # Evaluate
    # -----------------------------------------------------------------------
    results = {}

    if 'cls' in tasks:
        print('\n=== CLS Task ===')
        v_tara, v_qwen = vid_embs['cls']
        t_tara, t_qwen = txt_embs['cls']
        results['cls'] = eval_cls_ensemble(v_tara, v_qwen, t_tara, t_qwen, args.alpha)

    if 'ret' in tasks:
        print('\n=== RET Task ===')
        v_tara, v_qwen = vid_embs['ret']
        t_tara, t_qwen = txt_embs['ret']
        results['ret'] = eval_ret_ensemble(v_tara, v_qwen, t_tara, t_qwen, args.alpha)

    # -----------------------------------------------------------------------
    # Save + print summary
    # -----------------------------------------------------------------------
    out_dir = args.output_dir or os.path.join(_ROOT, '_ensemble_metrics')
    os.makedirs(out_dir, exist_ok=True)
    tag = (
        f'ensemble_{args.tara_model_name}_{args.qwen_model_name}'
        f'_a{args.alpha:g}_mmebv2_{args.task}'
    ).replace('.', 'p')
    save_path = os.path.join(out_dir, f'metrics_{tag}.json')
    with open(save_path, 'w') as f:
        json.dump({'alpha': args.alpha, 'results': results}, f, indent=4)
    print(f'\nSaved metrics → {save_path}')

    # Pretty summary
    print('\n' + '=' * 60)
    print(f'  alpha={args.alpha}  '
          f'TARA={args.tara_model_name}  Qwen={args.qwen_model_name}')
    print('=' * 60)
    if 'cls' in results:
        cls_r = results['cls']
        for ds, acc in cls_r['per_dataset'].items():
            print(f'  CLS  {ds:<14} {acc:.2f}')
        print(f'  CLS  {"MEAN":<14} {cls_r["mean"]:.2f}')
    if 'ret' in results:
        ret_r = results['ret']
        for ds, acc in ret_r['per_dataset'].items():
            print(f'  RET  {ds:<14} {acc:.2f}')
        print(f'  RET  {"MEAN":<14} {ret_r["mean"]:.2f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
