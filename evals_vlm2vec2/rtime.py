import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np

from shared.utils.log import tqdm_iterator
from shared.utils.io import load_json
from utils.general_retrieval_metrics import itm_eval


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
    txt2img = {i: i for i, k in enumerate(list(vid_fwd_emb))}
    img2txt = {i: i for i, k in enumerate(list(vid_fwd_emb))}
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
    txt2img = {i: i for i in range(len(ZT))}
    img2txt = {i: i for i in range(len(ZV))}
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


def embed_video(video_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
        videos=video_inputs,
        return_tensors="pt"
    )
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
    inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
    qry_output = model(qry=inputs)["qry_reps"]
    return qry_output.squeeze(0).float()


def embed_text(text):
    inputs = processor(
        text=text,
        images=None,
        return_tensors="pt"
    )
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    tgt_output = model(tgt=inputs)["tgt_reps"]
    return tgt_output.squeeze(0).float()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/work/piyush/pretrained_checkpoints/VLM2Vec-V2.0")
    parser.add_argument("--model_name", type=str, default="vlm2vec2")
    args = parser.parse_args()

    # Load RTime data
    data_dir = "/scratch/shared/beegfs/piyush/datasets/ReversedInTime"
    csv_path = f"{data_dir}/splits/all_meta.csv"
    df = pd.read_csv(csv_path)
    df = df[df.split == 'test']
    df.video_id = df.video_id.astype(str)
    data = load_json(f"{data_dir}/splits/test.json")

    filter_novel = False
    if filter_novel:
        from shared.utils.io import load_txt
        filter_ids = load_txt("testoftime_eval/rtime_non_matching_ids.txt")
        df = df[~df.video_id.isin(filter_ids)]
        data = {k: v for k, v in data.items() if k not in filter_ids}
        assert len(data) == len(df)
        print(f"Number of rows after filtering: {len(df)}")

    # Load VLM2Vec2 model
    repo_dir = "/users/piyush/projects/VLM2Vec"
    sys.path.append(repo_dir)

    from src.arguments import ModelArguments, DataArguments
    from src.model.model import MMEBModel
    from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
    from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info

    model_args = ModelArguments(
        model_name=args.model_path,
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )
    data_args = DataArguments()
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args)
    model = model.to('cuda', dtype=torch.bfloat16)
    model.eval()

    # Compute embeddings
    norm = lambda x: torch.nn.functional.normalize(x, dim=-1).cpu().float()
    vid_fwd_emb, vid_rev_emb = {}, {}
    cap_fwd_emb, cap_rev_emb = {}, {}

    for i in tqdm_iterator(range(len(df)), desc='Computing RTime embeddings'):
        row = df.iloc[i].to_dict()
        video_id = str(row['video_id'])

        try:
            vid_fwd_path = f"{data_dir}/videos/{video_id}.mp4"
            vid_rev_path = f"{data_dir}/videos/{video_id}-reverse.mp4"
            assert os.path.exists(vid_fwd_path)
            assert os.path.exists(vid_rev_path)

            cap_fwd_text = data[video_id]['forward_captions'][0]
            cap_rev_text = data[video_id]['reverse_captions'][0]

            with torch.no_grad():
                vid_fwd = norm(embed_video(vid_fwd_path))
                vid_rev = norm(embed_video(vid_rev_path))
                cap_fwd = norm(embed_text(cap_fwd_text))
                cap_rev = norm(embed_text(cap_rev_text))

            vid_fwd_emb[video_id] = vid_fwd
            vid_rev_emb[video_id] = vid_rev
            cap_fwd_emb[video_id] = cap_fwd
            cap_rev_emb[video_id] = cap_rev
        except:
            print(f"Error computing embeddings for {video_id}")
            continue

    df = df[df.video_id.isin(list(vid_fwd_emb.keys()))]
    print(f"Number of rows with valid embeddings: {len(df)}")

    # Save embeddings
    embeds = {
        "vid_fwd_emb": vid_fwd_emb,
        "vid_rev_emb": vid_rev_emb,
        "cap_fwd_emb": cap_fwd_emb,
        "cap_rev_emb": cap_rev_emb,
    }
    save_dir = os.path.join(args.model_path, "embs")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"embeddings_rtime_{args.model_name}.pt")
    torch.save(embeds, save_path)
    print(f"Saved embeddings to {save_path}")

    # Compute metrics
    metrics = {
        'binary': compute_rtime_binary_score(vid_fwd_emb, vid_rev_emb, cap_fwd_emb, cap_rev_emb),
        'origin': compute_rtime_origin_score(vid_fwd_emb, cap_fwd_emb),
        'hard': compute_rtime_hard_score(vid_fwd_emb, vid_rev_emb, cap_fwd_emb, cap_rev_emb),
    }
    print_metrics(metrics)

    # Save metrics
    result_dir = os.path.join(args.model_path, "metrics")
    os.makedirs(result_dir, exist_ok=True)
    suffix = "filtered" if filter_novel else "all"
    metrics_path = os.path.join(result_dir, f"metrics_rtime_{args.model_name}_{suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
