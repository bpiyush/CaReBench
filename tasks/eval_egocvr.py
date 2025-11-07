"""Evaluates the retrieval performance of the model on the EgoCVR dataset."""
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
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)
from utils.general_retrieval_metrics import itm_eval


PROMPT = "<video>\nEdit instruction: <sent>\n"\
         "Imagine the given text edit instruction applied on the given video.\n"\
         "Summarize the resulting video in one word:"
PROMPT = f"USER: {PROMPT} ASSISTANT: "
print(PROMPT)


def embed_video_text(encoder, video_path, edit_text, n_frames=8, verbose=False):
    generate_kwargs = {
        "max_new_tokens": 1,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }

    # Prepare video
    try:
        pixel_values = read_frames_decord(video_path, n_frames).unsqueeze(0)
    except:
        print(f"Error reading video: {video_path}. Returning random noise.")
        pixel_values = torch.randn(n_frames, 3, 270, 480)
    pixel_values = transform_pixel_values(pixel_values)
    nframes = pixel_values.shape[1]
    to_image = ToPILImage()
    batched_frames = []
    for batch in pixel_values:
        frames = [to_image(v) for v in batch]
        batched_frames.append(frames)

    for frames in batched_frames:

        # Video
        input_prompt = PROMPT.replace("<video>", "<image>"*len(frames))

        # Text
        input_prompt = input_prompt.replace('<sent>', edit_text)

        if verbose:
            print(input_prompt)

        input_ids = encoder.processor.get_text_inputs(input_prompt)
        frames = encoder.processor.get_pixel_values(frames)
        inputs = {
            "input_ids": input_ids,
            "pixel_values": frames
        }
        inputs = {k:v.to(encoder.model.device) for k,v in inputs.items() if v is not None}
        outputs = encoder.model.generate(
            **inputs,
            **generate_kwargs,
        )
        zv = outputs.hidden_states[0][-1][:, -1, :]
        break # Safe to break since it is just one video

    if verbose:
        print(zv.shape)

    return zv.squeeze(0)


if __name__ == "__main__":
    n_frames = 8
    
    # Load data
    data_dir = "/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/"
    meta_dir = f"{data_dir}/egocvr"
    video_dir = f"{meta_dir}/clips"
    df_base = pd.read_csv(f"{meta_dir}/egocvr_data.csv")
    df_anno = pd.read_csv(f"{meta_dir}/egocvr_annotations.csv")
    
    # Load model
    model_id = "/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint"
    encoder = AutoEncoder.from_pretrained(model_id, device_map='auto')
    su.misc.num_params(encoder.model)

    # Compute candidate embeddings
    candidate_clip_ids = []
    for x in df_anno.target_clip_ids.tolist():
        candidate_clip_ids.extend(eval(x))
    candidate_clip_ids = np.unique(candidate_clip_ids)
    candidates = {}
    for c in su.log.tqdm_iterator(candidate_clip_ids, desc='Computing candidate embeddings'):
        video_path = f"{video_dir}/{c}.mp4"
        if not os.path.exists(video_path):
            print(f"Target video does not exist: {c}. Skipping.")
            continue
        else:
            try:
                video_tensor = read_frames_decord(video_path, n_frames)
            except:
                print(f"Error reading video: {c}. ")
                video_tensor = torch.randn(n_frames, 3, 270, 480)
            with torch.no_grad():
                zv = encoder.encode_vision(video_tensor.unsqueeze(0)).cpu().squeeze(0).float()
                zv = torch.nn.functional.normalize(zv, dim=-1)
            candidates[c] = zv

    # Gather query embeddings
    queries = {}
    for i in su.log.tqdm_iterator(range(len(df_anno)), desc="Compute query embeddings"):
        row = df_anno.iloc[i].to_dict()
        video_path = f"{video_dir}/{row['video_clip_id']}.mp4"
        if not os.path.exists(video_path):
            print(f"Query video does not exist: {i}. Skipping.")
            continue
        edit_text = row['instruction']
        with torch.no_grad():
            zv = embed_video_text(encoder, video_path, edit_text, n_frames=8)
            zv = torch.nn.functional.normalize(zv, dim=-1)
            zv = zv.cpu().float()
        key = f"{edit_text}|{row['video_clip_id']}"
        queries[key] = zv
    import ipdb; ipdb.set_trace()

    # Compute metrics
    from collections import defaultdict
    import ipdb; ipdb.set_trace()
    zq = []
    zc = []
    txt2img = defaultdict(list)
    img2txt = defaultdict(list)
    j = 0
    indices = defaultdict(list)
    for i in range(len(df_anno)):
        row = df_anno.iloc[i].to_dict()
        query_key = f"{row['edit']}|{row['video_clip_id']}"
        if query_key not in queries:
            print(f"Missing query for {i}. Skipped.")
            continue
        
        for target_clip_id in eval(row['target_clip_ids']):
            if query_key not in queries or target_clip_id not in candidates:
                print(f"Missing value for {i}. Skipped.")
                continue
            zc.append(candidates[target_clip_id])
            zq.append(queries[query_key])
            indices[j].append(i)
            j += 1
    import ipdb; ipdb.set_trace()
    zq = torch.stack(zq).numpy()
    zc = torch.stack(zc).numpy()
    print(zq.shape, zc.shape)

    # i:q and t:c; and we care about q2c metrics, i.e., i2t, i.e., text_*
    score_q2c = zq @ zc.T
    score_c2q = zc @ zq.T
    indices = {i:i for i in range(len(score_q2c))}
    metrics = itm_eval(scores_i2t=score_q2c, scores_t2i=score_c2q, txt2img=indices, img2txt=indices, add_50=True)