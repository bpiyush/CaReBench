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

import shared.utils as su
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)
from models.modeling_encoders import AutoEncoder
from utils.video import read_frames_decord
from models.modeling_encoders import AutoEncoder


split_to_filename = {
    'count': 'count-00000-of-00001-b1a5fcd07825dea8.parquet',
    'rotation': 'rotation-00000-of-00001-08423ae1066428bd.parquet',
    'velocity_frequency': 'velocity_frequency-00000-of-00001-d77a9ec4957bf2b4.parquet',
    'direction': 'direction-00000-of-00001-6ad53d19ff32f95f.parquet',
    'shape_trend': 'shape_trend-00000-of-00001-0d46b5472272272a.parquet',
    'visual_cues': 'visual_cues-00000-of-00001-d4585588d5b20c3d.parquet',
}


def format_question(row):
    q = row['question']
    options = row['options']
    # try:
    #     answer = list(options).index(str(row['answer']))
    # except:
    #     print("Answer {row['answer']} not in options {options}. Picking a random answer.")
    #     import random
    #     import ipdb; ipdb.set_trace()
    #     # Pick a random answer
    #     answer = random.randint(0, len(options) - 1)
    answer_index = row['answer']
    
    # Add options with letters
    options = [f"{chr(65+i)}: {o}" for i, o in enumerate(options)]
    answer_letter = chr(65 + answer_index)
    options = '\n'.join(options)
    q = f"{q}\nOptions: \n{options} \nPick the best suited option."
    return q, answer_letter


def generate_answer(d, n_frames=4, verbose=False):
    video_path = f"{data_dir}/{d['video_name']}"
    
    
    prompt_template = f"<video>\n<question>\nAnswer: ("
    try:
        frames_raw = read_frames_decord(video_path, n_frames)
    except:
        print("Error reading frames for video: ", video_path)
        return "<video_read_error>", d['answer_formatted']
    prompt = prompt_template.replace("<question>", d['question_formatted'])
    true_answer = d['answer_formatted']
    
    if verbose:
        print(prompt)
        su.visualize.concat_images_with_border(
            [PIL.Image.fromarray(f.numpy()) for f in frames_raw.permute((0, 2, 3, 1))]
        ).save("frames.png")

    try:
    
        with torch.no_grad():
        
            pixel_values = transform_pixel_values(frames_raw) # [B, T, C, H, W]
            
            to_image = ToPILImage()
            batched_frames = []
            for batch in pixel_values:
                frames = [to_image(v) for v in batch]
                batched_frames.append(frames)
        
            generate_kwargs = {
                "max_new_tokens": 1,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
                "top_p": 1.,
                "temperature": 0.,
                "do_sample": False,
            }
        
            for frames in batched_frames:
                input_prompt = prompt.replace("<video>", "<image>"*len(frames))
                input_ids = model_base.processor.get_text_inputs(input_prompt)
                frames = model_base.processor.get_pixel_values(frames)
                inputs = {
                    "input_ids": input_ids,
                    "pixel_values": frames
                }
                inputs = {k:v.to(model_base.model.device) for k,v in inputs.items() if v is not None}
                outputs = model_base.model.generate(
                    **inputs,
                    **generate_kwargs,
                )
                output_text = model_base.processor.tokenizer.decode(
                    outputs[0][0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True,
                )
                if verbose:
                    print("Generated answer: ", output_text)
                    print("Actual answer: ", true_answer)
                return output_text, true_answer
    except:
        print("Error generating answer for video: ", video_path)
        return "<error>", true_answer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/work/piyush/pretrained_checkpoints/TARA')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--n_frames', type=int, default=8)
    parser.add_argument("--split", type=str, default="count")
    args = parser.parse_args()
    
    data_dir = "/scratch/shared/beegfs/piyush/datasets/TOMATO/"
    data_file = f'{data_dir}/data/{split_to_filename[args.split]}'
    data = pd.read_parquet(data_file)
    data = data.to_dict(orient='records')
    data_name = args.split
    
    # Add question and answer to data
    for d in su.log.tqdm_iterator(data, desc='Formatting questions'):
        q, answer_letter = format_question(d)
        d['question_formatted'] = q
        d['answer_formatted'] = answer_letter
        d['video_name'] = f"videos/{d['demonstration_type']}/{d['key']}.mp4"


    # Load model
    model_path = args.model_path
    device_map = "auto"
    attn_implementation = 'flash_attention_2'
    model_base = AutoEncoder.from_pretrained(model_path, device_map=device_map, attn_implementation=attn_implementation, use_flash_attn=False, dtype=torch.bfloat16)
    su.misc.num_params(model_base.model)


    from copy import deepcopy
    iterator = su.log.tqdm_iterator(range(len(data)), desc='Generating answers')
    debug = args.debug
    dry_run = args.dry_run
    df_out = []
    for i in iterator:
        d = deepcopy(data[i])
        a_pred, a_true = generate_answer(d, n_frames=args.n_frames, verbose=debug)
        df_out.append(
            dict(a_pred=a_pred, a_true=a_true, **d)
        )
        if debug:
            break

        if dry_run and i == 5:
            break
    
    if not debug:
        df_out = pd.DataFrame(df_out)
        accu = (df_out['a_pred'] == df_out['a_true']).mean()
        print(f"Accuracy: {accu*100:.2f}")
        
        # Save results
        model_name = model_path.split("/")[-1]
        os.makedirs("outputs", exist_ok=True)
        df_out.to_csv(f"outputs/tomato_results-{model_name}-{data_name}.csv", index=False)
        print(f"Results saved to outputs/tomato_results-{model_name}-{data_name}.csv")