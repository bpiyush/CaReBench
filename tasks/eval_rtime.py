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
from utils.general_retrieval_metrics import itm_eval


def compute_video_embeddings(df, vfc, vp, data_dir):
    vid_fwd_emb = {}
    vid_rev_emb = {}
    for i in su.log.tqdm_iterator(range(len(df))):
        row = df.iloc[i].to_dict()
        vid_fwd = f"{data_dir}/videos/{row['video_id']}.mp4"
        
        try:

            vid_fwd_tensor = vp(vid_fwd)
            vid_rev_tensor = torch.flip(vid_fwd_tensor, dims=(0,))

            zv = vfc(vid_fwd_tensor)
            zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float()
            vid_fwd_emb[str(row['video_id'])] = zv

            zv = vfc(vid_rev_tensor)
            zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float()
            vid_rev_emb[str(row['video_id'])] = zv
        
        except:
            print(f"Error computing video embedding for {vid_fwd}")
            continue
    return vid_fwd_emb, vid_rev_emb


def compute_text_embeddings(df, tfc, data):
    cap_fwd_emb = {}
    cap_rev_emb = {}
    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing text features'):
        row = df.iloc[i].to_dict()
        video_id = str(row['video_id'])
        x = data[video_id]

        zt = tfc(x['forward_captions'][0])
        zt = torch.nn.functional.normalize(zt, dim=-1).cpu().float()
        cap_fwd_emb[video_id] = zt

        zt = tfc(x['reverse_captions'][0])
        zt = torch.nn.functional.normalize(zt, dim=-1).cpu().float()
        cap_rev_emb[video_id] = zt
    return cap_fwd_emb, cap_rev_emb


def compute_rtime_binary_score(vid_fwd_emb, vid_rev_emb, cap_fwd_emb, cap_rev_emb):
    t2v_acc = []
    v2t_acc = []
    for k in vid_fwd_emb:
        zv_fwd = vid_fwd_emb[k]
        zv_rev = vid_rev_emb[k]
        zt_fwd = cap_fwd_emb[str(k)]
        zt_rev = cap_rev_emb[str(k)]
        sim = torch.stack([zv_fwd, zv_rev]) @ torch.stack([zt_fwd, zt_rev]).T
        t2v_acc.append(sim[0, 0] > sim[1, 0])
        v2t_acc.append(sim[0, 0] > sim[0, 1])
    t2v_acc = np.mean(t2v_acc).item()
    v2t_acc = np.mean(v2t_acc).item()
    return dict(t2v_acc=t2v_acc, v2t_acc=v2t_acc)


def compute_rtime_origin_score(vid_fwd_emb, cap_fwd_emb):
    ZV = torch.stack([vid_fwd_emb[k] for k in vid_fwd_emb])
    ZT = torch.stack([cap_fwd_emb[str(k)] for k in vid_fwd_emb])

    scores_i2t = (ZV @ ZT.T).numpy()
    scores_t2i = (ZT @ ZV.T).numpy()
    scores_i2t.shape, scores_t2i.shape

    txt2img = {i:i for i, k in enumerate(list(vid_fwd_emb))}
    img2txt = {i:i for i, k in enumerate(list(vid_fwd_emb))}

    metrics = itm_eval(scores_i2t, scores_t2i, txt2img, img2txt)
    metrics = {k: v.item() for k, v in metrics.items()}
    return metrics


def compute_rtime_hard_score(vid_fwd_emb, vid_rev_emb, cap_fwd_emb, cap_rev_emb):
    ZV_fwd = torch.stack([vid_fwd_emb[k] for k in vid_fwd_emb])
    ZV_rev = torch.stack([vid_rev_emb[k] for k in vid_fwd_emb])
    ZT_fwd = torch.stack([cap_fwd_emb[str(k)] for k in vid_fwd_emb])
    ZT_rev = torch.stack([cap_rev_emb[str(k)] for k in vid_fwd_emb])
    ZV = torch.cat([ZV_fwd, ZV_rev], dim=0)
    ZT = torch.cat([ZT_fwd, ZT_rev], dim=0)
    scores_i2t = (ZV @ ZT.T).numpy()
    scores_t2i = (ZT @ ZV.T).numpy()
    txt2img = {i:i for i in range(len(ZT))}
    img2txt = {i:i for i in range(len(ZV))}
    metrics = itm_eval(scores_i2t, scores_t2i, txt2img, img2txt)
    metrics = {k: v.item() for k, v in metrics.items()}
    return metrics


def print_metrics(metrics):
    _list = []
    for z in ['origin', 'hard']:
        for y in ['img', 'txt']:
            for x in ['r1', 'r5', 'r10']:
                _list.append(np.round(metrics[z][f"{y}_{x}"]))
    _list.append(np.round(100. * metrics['binary']['t2v_acc'], 1))
    _list.append(np.round(100. * metrics['binary']['v2t_acc'], 1))
    _list = np.array(_list).astype(str)
    print(' & '.join(_list))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/work/piyush/pretrained_checkpoints/Qwen2-VL-7B-Instruct")
    parser.add_argument("--model_name", type=str, default="qwen2vl7b")
    args = parser.parse_args()
    
    data_dir = "/scratch/shared/beegfs/piyush/datasets/ReversedInTime"
    csv_path = f"{data_dir}/splits/all_meta.csv"
    df = pd.read_csv(csv_path)
    df = df[df.split == 'test']
    df.video_id = df.video_id.astype(str)
    data = su.io.load_json(f"{data_dir}/splits/test.json")
    
    
    from notebooks.eval_care_retrieval import load_model
    # model_path = "/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint"
    # model_name = "tarsier7b+tara"
    # model_path = "/work/piyush/pretrained_checkpoints/Tarsier-7b"
    # model_name = "tarsier7b"
    # model_path = "/work/piyush/pretrained_checkpoints/Qwen2-VL-7B-Instruct"
    # model_name = "qwen2vl7b"
    model_path = args.model_path
    model_name = args.model_name
    vfc, tfc, vp  = load_model(_id=model_path)
    
    vid_fwd_emb, vid_rev_emb = compute_video_embeddings(df, vfc, vp, data_dir)
    cap_fwd_emb, cap_rev_emb = compute_text_embeddings(df, tfc, data)
    
    # Only keep the rows with valid video embeddings
    df = df[df.video_id.isin(list(vid_fwd_emb.keys()))]
    print(f"Number of rows with valid video embeddings: {len(df)}")
    
    # Compute all metrics
    metrics = {
        'binary': compute_rtime_binary_score(vid_fwd_emb, vid_rev_emb, cap_fwd_emb, cap_rev_emb),
        'origin': compute_rtime_origin_score(vid_fwd_emb, cap_fwd_emb),
        'hard': compute_rtime_hard_score(vid_fwd_emb, vid_rev_emb, cap_fwd_emb, cap_rev_emb),
    }
    # print(json.dumps(metrics, indent=4))
    print_metrics(metrics)
    
    # Save metrics
    result_dir = "./results"
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"metrics_rtime_{model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

