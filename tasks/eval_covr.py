import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import json
from torch.nn.functional import cosine_similarity
from utils.video import read_frames_decord
from IPython.display import display, Markdown, Latex

import shared.utils as su
from notebooks.eval_care_retrieval import load_model
from utils.video import read_frames_decord
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
    pixel_values = read_frames_decord(video_path, n_frames).unsqueeze(0)
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint")
    args = parser.parse_args()
    model_id = args.model_id
    
    
    from models.modeling_encoders import AutoEncoder
    encoder = AutoEncoder.from_pretrained(model_id, device_map='auto')
    su.misc.num_params(encoder.model)
    
    data_dir = "/scratch/shared/beegfs/piyush/datasets/WebVid-CoVR"
    video_dir = '/datasets/WebVid/videos'
    df = pd.read_csv(f"{data_dir}/webvid8m-covr_test-cleaned.csv")
    print("Number of rows in CoVR-test: ", len(df))
    
    # Compute video embeddings for the candidate videos
    videos = set(df.video2.tolist())
    candidates = {}
    n_frames = 16
    for video in su.log.tqdm_iterator(videos, desc='Computing features for candidate videos'):
        video_path = f"{video_dir}/{video}"
        assert os.path.exists(video_path)
        video_tensor = read_frames_decord(video_path, n_frames)
        with torch.no_grad():
            zv = encoder.encode_vision(video_tensor.unsqueeze(0)).cpu().squeeze(0).float()
            zv = torch.nn.functional.normalize(zv, dim=-1)
        candidates[video] = zv

    # Gather query embeddings
    query_embeds = {}
    for i in su.log.tqdm_iterator(range(len(query_embeds), len(df)), desc="Compute query embeddings"):
        row = df.iloc[i].to_dict()
        video_path = f"{video_dir}/{row['video1']}"
        edit_text = row['edit']
        with torch.no_grad():
            try:
                zv = embed_video_text(encoder, video_path, edit_text, n_frames=n_frames)
                zv = torch.nn.functional.normalize(zv, dim=-1)
                zv = zv.cpu().float()
                key = f"{edit_text}|{row['video1']}"
                query_embeds[key] = zv
            except:
                print(f"Skpping {i}")
                continue

    zq = []
    zc = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        query_key = f"{row['edit']}|{row['video1']}"
        candi_key = row['video2']
        if query_key not in query_embeds or candi_key not in candidates:
            print(f"Missing value for {i}. Skipped.")
            continue
        zq.append(query_embeds[query_key])
        zc.append(candidates[candi_key])
    zq = torch.stack(zq).numpy()
    zc = torch.stack(zc).numpy()
    print(zq.shape, zc.shape)

    # i:q and t:c; and we care about q2c metrics, i.e., i2t, i.e., text_*
    score_q2c = zq @ zc.T
    score_c2q = zc @ zq.T
    indices = {i:i for i in range(len(score_q2c))}
    metrics = itm_eval(
        scores_i2t=score_q2c,
        scores_t2i=score_c2q,
        txt2img=indices,
        img2txt=indices,
        add_50=True,
    )
    print(metrics)