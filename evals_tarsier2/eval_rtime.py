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
    parser.add_argument("--model_path", type=str, default="/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/")
    parser.add_argument("--model_name", type=str, default="tarsier2_7b")
    args = parser.parse_args()
    
    data_dir = "/scratch/shared/beegfs/piyush/datasets/ReversedInTime"
    csv_path = f"{data_dir}/splits/all_meta.csv"
    df = pd.read_csv(csv_path)
    df = df[df.split == 'test']
    df.video_id = df.video_id.astype(str)
    data = su.io.load_json(f"{data_dir}/splits/test.json")
    
    # Filter to only keep IDs that have verbs substantially different from the our Ego4D set.
    filter_novel = False
    if filter_novel:
        filter_ids = su.io.load_txt("testoftime_eval/rtime_non_matching_ids.txt")
        df = df[~df.video_id.isin(filter_ids)]
        data = {k: v for k, v in data.items() if k not in filter_ids}
        assert len(data) == len(df)
        print(f"Number of rows after filtering: {len(df)}")
    
    # Load model
    from models.modeling_encoders import AutoEncoder
    model = AutoEncoder.from_pretrained(
        args.model_path,
        device_map='auto',
        attn_implementation='flash_attention_2',
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)
    import ipdb; ipdb.set_trace()
    
    vid_fwd_emb, vid_rev_emb = {}, {}
    cap_fwd_emb, cap_rev_emb = {}, {}
    norm = lambda x: torch.nn.functional.normalize(x, dim=-1).cpu().float().squeeze(0)
    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing video embeddings'):
        row = df.iloc[i].to_dict()
        
        video_id = str(row['video_id'])
        
        try:
            vid_fwd_path = f"{data_dir}/videos/{video_id}.mp4"
            vid_rev_path = f"{data_dir}/videos/{video_id}-reverse.mp4"
            assert os.path.exists(vid_fwd_path)
            assert os.path.exists(vid_rev_path)
            
            cap_fwd = data[video_id]['forward_captions'][0]
            cap_rev = data[video_id]['reverse_captions'][0]
            
            with torch.no_grad():
                vid_fwd = model.encode_vision(vid_fwd_path)
                vid_fwd = norm(vid_fwd)
                vid_rev = model.encode_vision(vid_rev_path)
                vid_rev = norm(vid_rev)
                cap_fwd = model.encode_text(cap_fwd)
                cap_fwd = norm(cap_fwd)
                cap_rev = model.encode_text(cap_rev)
                cap_rev = norm(cap_rev)

            vid_fwd_emb[video_id] = vid_fwd
            vid_rev_emb[video_id] = vid_rev
            cap_fwd_emb[video_id] = cap_fwd
            cap_rev_emb[video_id] = cap_rev
        except:
            print(f"Error computing embeddings for {video_id}")
            continue
    import ipdb; ipdb.set_trace()
        
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
    result_dir = os.path.join(args.model_path, "metrics")
    os.makedirs(result_dir, exist_ok=True)
    suffix = "filtered" if filter_novel else "all"
    with open(os.path.join(result_dir, f"metrics_rtime_{args.model_name}_{suffix}.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    
    # from notebooks.eval_care_retrieval import load_model
    # # model_path = "/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint"
    # # model_name = "tarsier7b+tara"
    # # model_path = "/work/piyush/pretrained_checkpoints/Tarsier-7b"
    # # model_name = "tarsier7b"
    # # model_path = "/work/piyush/pretrained_checkpoints/Qwen2-VL-7B-Instruct"
    # # model_name = "qwen2vl7b"
    # model_path = args.model_path
    # model_name = args.model_name
    # vfc, tfc, vp  = load_model(_id=model_path)
    
    # vid_fwd_emb, vid_rev_emb = compute_video_embeddings(df, vfc, vp, data_dir)
    # cap_fwd_emb, cap_rev_emb = compute_text_embeddings(df, tfc, data)
    
    # # Only keep the rows with valid video embeddings
    # df = df[df.video_id.isin(list(vid_fwd_emb.keys()))]
    # print(f"Number of rows with valid video embeddings: {len(df)}")
    
    # # Compute all metrics
    # metrics = {
    #     'binary': compute_rtime_binary_score(vid_fwd_emb, vid_rev_emb, cap_fwd_emb, cap_rev_emb),
    #     'origin': compute_rtime_origin_score(vid_fwd_emb, cap_fwd_emb),
    #     'hard': compute_rtime_hard_score(vid_fwd_emb, vid_rev_emb, cap_fwd_emb, cap_rev_emb),
    # }
    # # print(json.dumps(metrics, indent=4))
    # print_metrics(metrics)
    
    # # Save metrics
    # result_dir = "./results"
    # os.makedirs(result_dir, exist_ok=True)
    # with open(os.path.join(result_dir, f"metrics_rtime_{model_name}-filtered.json"), "w") as f:
    #     json.dump(metrics, f, indent=4)

