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
from datasets import load_dataset


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_id', type=str, default='/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint')
parser.add_argument('--model_name', type=str, default='tarsier7b+tara')
args = parser.parse_args()


# Load model
n_frames = 8
model_id = args.model_id
encoder = AutoEncoder.from_pretrained(model_id, device_map='auto')
su.misc.num_params(encoder.model)


def gather_text_embeddings(texts):
    ZT = {}
    for text in su.log.tqdm_iterator(texts, desc='Computing text embeddings'):
        with torch.no_grad():
            zt = encoder.encode_text(text)
            zt = torch.nn.functional.normalize(zt, dim=-1).squeeze(0).cpu().float()
        ZT[text] = zt
    return ZT


# Video classification
data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
meta_config = su.io.load_yml(
    '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
)
# Load video embeddings (should be pre-computed)
path = f"{data_root}/features/{args.model_name}_video_embeddings_mmebv2_video_cls.pt"
assert os.path.exists(path)
video_embs = torch.load(path)


# SSv2
ds_key = "SmthSmthV2"
su.log.print_update(f"Processing {ds_key}")
d = meta_config[ds_key]
eval_type = d['eval_type']
data = su.io.load_jsonl(f"{data_root}/video-tasks/data/{d['json_name']}")
data = pd.DataFrame(data)
all_texts = np.unique(data.neg_text.sum())
text_to_emb = gather_text_embeddings(all_texts)
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
    text_to_emb_local = gather_text_embeddings(texts_local)
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

# Video retrieval
data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
meta_config = su.io.load_yml(
    '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml'
)
# Load video embeddings (should be pre-computed)
path = f"{data_root}/features/{args.model_name}_video_embeddings_mmebv2_video_ret.pt"
assert os.path.exists(path)
video_embs = torch.load(path)
# This defines the huggingface repo and subset for each dataset
# (repo, subset, split)
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
text_embeds = {}
for ds_key in meta_config:
    su.log.print_update(f"Processing {ds_key}")
    d = meta_config[ds_key]
    repo, subset, split = json_paths[ds_key]
    df = pd.DataFrame(load_dataset(repo, subset)[split])
    video_dir = f"{video_root}/{ds_key}/frames"
    video_ids = os.listdir(video_dir)
    assert len(video_ids) == len(df)
    df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)
    print(json.dumps(df.iloc[0].to_dict(), indent=2))

    all_texts = [
        captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))
    ]
    all_texts = np.unique(np.concatenate(all_texts))
    print("Total number of next captions: ", len(all_texts))
    text_embeds[ds_key] = gather_text_embeddings(all_texts)
    su.log.print_update(f"")


ret_accs = {}
for ds_key in meta_config:
    su.log.print_update(f"Processing {ds_key}")
    d = meta_config[ds_key]

    repo, subset, split = json_paths[ds_key]
    df = pd.DataFrame(load_dataset(repo, subset)[split])
    video_dir = f"{video_root}/{ds_key}/frames"
    video_ids = os.listdir(video_dir)
    assert len(video_ids) == len(df)
    df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)

    print(json.dumps(df.iloc[0].to_dict(), indent=2))

    all_texts = [
        captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))
    ]
    all_texts = np.unique(np.concatenate(all_texts))
    print("Total number of next captions: ", len(all_texts))
    text_emb = text_embeds[ds_key]

    zv = torch.stack([video_embs[c] for c in df.video_id.tolist()])
    zt = torch.stack([text_emb[t] for t in all_texts])

    sim = zv @ zt.T
    pred_indices = sim.argmax(dim=-1)
    pred_captions = np.array([all_texts[i] for i in pred_indices])
    actu_captions = [captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))]
    is_correct = [int(x in y) for x, y in zip(pred_captions, actu_captions)]
    accuracy = np.round(np.mean(is_correct) * 100., 2).item()
    ret_accs[ds_key] = accuracy
    su.log.print_update(f"")

mean_acc = np.mean([v for k, v in ret_accs.items()])
mean_acc = np.round(mean_acc, 2)
print(f"Mean accuracy: {mean_acc:.2f}")
print(ret_accs)
import ipdb; ipdb.set_trace()