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
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    functional,
)
from torchvision.transforms.functional import InterpolationMode
from utils.general_retrieval_metrics import itm_eval


PROMPT = "<video>\nEdit instruction: <sent>\n"\
         "Imagine the given text edit instruction applied on the given video.\n"\
         "Summarize the resulting video in one word:"
PROMPT = f"USER: {PROMPT} ASSISTANT: "
print(PROMPT)


def embed_video_text(encoder, video_path, edit_text, n_frames=8, verbose=False):
    
    prompt = PROMPT
    prompt = prompt.replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")

    # Prepare video
    pixel_values = read_frames_decord(video_path, n_frames)
    batched_pixel_values = transform_pixel_values(pixel_values)

    VIDEO_MIN_PIXELS = 128 * 28 * 28
    VIDEO_MAX_PIXELS = 768 * 28 * 28
    VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
    FRAME_FACTOR = 2
    IMAGE_FACTOR = 28
    
    for pixel_values in batched_pixel_values:
        nframes, _, height, width = pixel_values.shape

        min_pixels = VIDEO_MIN_PIXELS
        total_pixels = VIDEO_TOTAL_PIXELS
        max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
        max_pixels = 230400
        resized_height, resized_width = encoder.smart_resize(
            height,
            width,
            factor=IMAGE_FACTOR,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        pixel_values = functional.resize(
            pixel_values,
            [resized_height, resized_width],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ).float()
        
        inputs = encoder.processor(
            text=[prompt],
            images=None,
            videos=[pixel_values],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(encoder.model.device)
        with torch.inference_mode():
            output = encoder.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
        zv = output.hidden_states[0][-1][:, -1, :]
        break # Safe to break since it is just one video
        
    if verbose:
        print(zv.shape)
    return zv.squeeze(0).float()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint")
    parser.add_argument("--model_name", type=str, default="tarsier7b")
    parser.add_argument("--n_frames", type=int, default=16)
    args = parser.parse_args()
    model_id = args.model_id
    model_name = args.model_name
    
    
    from models.modeling_encoders import AutoEncoder
    encoder = AutoEncoder.from_pretrained(model_id, device_map='auto')
    su.misc.num_params(encoder.model)
    
    data_dir = "/scratch/shared/beegfs/piyush/datasets/WebVid-CoVR"
    video_dir = '/datasets/WebVid/videos'
    df = pd.read_csv(f"{data_dir}/webvid8m-covr_test-cleaned.csv")
    print("Number of rows in CoVR-test: ", len(df))

    # Gather query embeddings
    query_embeds = {}
    for i in su.log.tqdm_iterator(range(len(query_embeds), len(df)), desc="Compute query embeddings"):
        row = df.iloc[i].to_dict()
        video_path = f"{video_dir}/{row['video1']}"
        edit_text = row['edit']
        with torch.no_grad():
            try:
                zv = embed_video_text(encoder, video_path, edit_text, n_frames=args.n_frames)
                zv = torch.nn.functional.normalize(zv, dim=-1)
                zv = zv.cpu().float()
                key = f"{edit_text}|{row['video1']}"
                query_embeds[key] = zv
            except:
                print(f"Skpping {i}")
                continue

    # Compute video embeddings for the candidate videos
    videos = set(df.video2.tolist())
    candidates = {}
    n_frames = args.n_frames
    for video in su.log.tqdm_iterator(videos, desc='Computing features for candidate videos'):
        video_path = f"{video_dir}/{video}"
        assert os.path.exists(video_path)
        video_tensor = read_frames_decord(video_path, n_frames)
        with torch.no_grad():
            zv = encoder.encode_vision(video_tensor.unsqueeze(0)).cpu().squeeze(0).float()
            zv = torch.nn.functional.normalize(zv, dim=-1)
        candidates[video] = zv

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
    
    # Save metrics
    result_dir = f"{model_id}/metrics"
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"metrics_covr_{model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)