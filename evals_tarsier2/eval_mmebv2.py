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


def gather_text_embeddings(encoder, texts):
    ZT = {}
    for text in su.log.tqdm_iterator(texts, desc='Computing text embeddings'):
        with torch.no_grad():
            zt = encoder.encode_text(text)
            zt = torch.nn.functional.normalize(zt, dim=-1).squeeze(0).cpu().float()
        ZT[text] = zt
    return ZT


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/')
    parser.add_argument('--model_name', type=str, default='tarsier2_7b')
    parser.add_argument("--task", type=str, default='cls', choices=['cls', 'ret'])
    args = parser.parse_args()

    # Load model
    model = AutoEncoder.from_pretrained(
        args.model_path,
        device_map='auto',
        attn_implementation='flash_attention_2',
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)
    
    
    # Load video embeddings
    feat_dir = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/features"
    feat_path = f"{feat_dir}/{args.model_name}_video_embeddings_mmebv2_video_{args.task}.pt"
    assert os.path.exists(feat_path)
    video_embs = torch.load(feat_path)


    # [CLS] Video classification
    data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
    meta_config = su.io.load_yml(
        '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
    )

    # SSv2
    ds_key = "SmthSmthV2"
    su.log.print_update(f"Processing {ds_key}")
    d = meta_config[ds_key]
    eval_type = d['eval_type']
    data = su.io.load_jsonl(f"{data_root}/video-tasks/data/{d['json_name']}")
    data = pd.DataFrame(data)
    all_texts = np.unique(data.neg_text.sum())
    text_to_emb = gather_text_embeddings(model, all_texts)
    correct = []
    for j in su.log.tqdm_iterator(range(len(data)), desc='Gathering predictions'):
        row = data.iloc[j].to_dict()
        texts = row['neg_text']
        zt = torch.stack([text_to_emb[t] for t in texts])
        gt_index = texts.index(row['pos_text'])
        sim = video_embs[row['video_id']] @ zt.T
        pred_index = sim.argmax().item()
        correct.append(int(gt_index == pred_index))
    accuracy = np.mean(correct)
    accuracies = {'SmthSmthV2': np.round(accuracy * 100, 2)}

    # Other datasets
    for ds_key in ['HMDB51', 'UCF101', 'K700', 'Breakfast']:
        su.log.print_update(f"Processing {ds_key}")
        d = meta_config[ds_key]
        eval_type = d['eval_type']
        data = su.io.load_jsonl(f"{data_root}/video-tasks/data/{d['json_name']}")
        data = pd.DataFrame(data)
        print("Number of rows: ", len(data))

        # Only keep those rows for which video embedding texts
        data = data[data.video_id.apply(lambda x: x in set(video_embs))]
        print("Number of rows after filtering: ", len(data))

        zv = torch.stack([video_embs[c] for c in data.video_id.tolist()])
        texts_local = data.pos_text.unique()
        text_to_emb_local = gather_text_embeddings(model, texts_local)
        zt = torch.stack([text_to_emb_local[c] for c in data.pos_text.tolist()])

        sim = zv @ zt.T
        pred_indices = sim.argmax(dim=-1)
        pred_classes = [data.pos_text.tolist()[i] for i in pred_indices]
        accuracy = np.round((np.array(pred_classes) == np.array(data.pos_text)).mean() * 100, 2)
        accuracies[ds_key] = accuracy
        
        su.log.print_update(f"")
    mean_accuracy = np.mean([v for v in accuracies.values()])
    print(f"Mean accuracy: {mean_accuracy:.2f}")
    print(accuracies)
