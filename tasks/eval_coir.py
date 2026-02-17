"""Evaluates composed image retrieval on MMEB subsets."""
import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import decord
import json
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import PIL.Image
from glob import glob
from natsort import natsorted

from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict as edict
import numpy as np
import json
import einops
from collections import defaultdict

import shared.utils as su
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)
from models.modeling_encoders import AutoEncoder
from notebooks.eval_care_retrieval import load_model, load_data
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
    functional,
)
from datasets import load_dataset


def read_image(image_path):
    image = PIL.Image.open(image_path)
    image = torch.tensor(np.asarray(image)).unsqueeze(0)
    image = image.permute((0, 3, 1, 2))
    return image


PROMPT = "Source image: <video>\nEdit instruction: <sent>\n"\
        "Look at the attached image carefully. The provided text is instruction to edit the image to a new sentence. "\
        "Imagine this edit instruction being applied to the provided image.\n"\
        "Summarize the resulting edited image in one word:"
PROMPT = f"USER: {PROMPT} ASSISTANT: "


def embed_image_text(encoder, image_path, edit_text, verbose=False):
    generate_kwargs = {
        "max_new_tokens": 1,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }

    # Prepare image
    pixel_values = read_image(image_path)
    pixel_values = transform_pixel_values(pixel_values)
    # print(pixel_values.shape)
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
    parser.add_argument("--datasets", type=str, default='["FashionIQ", "CIRR"]')
    parser.add_argument("--model_paths", type=str, default='{"Tarsier (Base Model)": "/work/piyush/pretrained_checkpoints/Tarsier-7b", "TARA (CoVR+CiA)": "/work/piyush/experiments/CaRe/Tarsier-7b/covr/chiral10k-covr10k/merged_checkpoint"}')
    args = parser.parse_args()

    model_paths = json.loads(args.model_paths)
    print(f"Model paths: {model_paths}")
    
    
    # Loop over datasets
    for ds_name in eval(args.datasets):
        print(f"Processing dataset: {ds_name}")

        # Load data
        image_dir = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/image-tasks/MMEB/"
        data = load_dataset("/scratch/shared/beegfs/piyush/datasets/MMEB-eval", ds_name)['test']
        tgt_paths = []
        for d in su.log.tqdm_iterator(data):
            tgt_image_paths = [f"{image_dir}/{f}" for f in d['tgt_img_path']]
            tgt_paths.extend(tgt_image_paths)
        tgt_paths = np.unique(tgt_paths)
        print(f"Number of target images: {len(tgt_paths)}")

        device_map = "auto"
        attn_implementation = 'flash_attention_2'

        models = {}
        for key, model_path in model_paths.items():
            model = AutoEncoder.from_pretrained(
                model_path,
                device_map=device_map,
                attn_implementation=attn_implementation,
                use_flash_attn=False,
                dtype=torch.bfloat16,
            )
            models[key] = model
            su.misc.num_params(model.model)
            print("-" * 100)
        
        
        # Run inference
        for model_key in models.keys():
            model = models[model_key]
            
            # Compute target image embeddings
            tgt_embeds = {}
            for f in su.log.tqdm_iterator(tgt_paths, desc="Computing embeddings for target images"):
                with torch.no_grad():
                    pixel_values = read_image(f)
                    zv = model.encode_vision(pixel_values).squeeze(0).cpu().float()
                tgt_embeds[f] = zv
            
            # Compute query image embeddings and compute similarity scores
            correct = []
            for d in su.log.tqdm_iterator(data):
                caption = d['qry_text'].split(": ")[-1].strip("\n")
                query_image_path = f"{image_dir}/{d['qry_img_path']}"
                with torch.no_grad():
                    zq = embed_image_text(model, query_image_path, caption).cpu().float()
                    zq = torch.nn.functional.normalize(zq, dim=-1)
                
                # Remove same image from target images
                tgt_image_paths = [f"{image_dir}/{f}" for f in d['tgt_img_path'] if f != d['qry_img_path']]
                # tgt_image_paths = [f"{image_dir}/{f}" for f in d['tgt_img_path']]
                
                # Remove same image from target images
                # import ipdb; ipdb.set_trace()

                zt = torch.stack([tgt_embeds[f] for f in tgt_image_paths])
                zt = torch.nn.functional.normalize(zt, dim=-1)
                sim = zq @ zt.T
                correct.append(torch.argmax(sim) == 0)
            correct = torch.tensor(correct).float()
            accuracy = torch.mean(correct)
            print(f"Accuracy for {model_key} on {ds_name}: {accuracy*100.:.2f}")