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


# generate_kwargs = {
#     "do_sample": False,
#     "max_new_tokens": 512,
#     "top_p": 1.,
#     "temperature": 0.,
#     "use_cache": True
# }


def generate_answer(d, n_frames=4, verbose=False):
    video_path = f"{data_dir}/{d['video_name']}"
    
    
    prompt_template = f"<video>\n<question>\nAnswer: ("
    try:
        frames_raw = read_frames_decord(video_path, n_frames)
    except:
        print("Error reading frames for video: ", video_path)
        return "<video_read_error>", d['ans']
    prompt = prompt_template.replace("<question>", d['question'])
    true_answer = d['ans']
    
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
    parser.add_argument("--split", type=str, default="UCF101")
    args = parser.parse_args()
    
    data_dir = "/scratch/shared/beegfs/piyush/datasets/AoTBench"
    # import ipdb; ipdb.set_trace()
    # files = glob(f"{data_dir}/AoTBench/data_files/*.json")
    # Just run on UCF file for now
    data_file = f'/scratch/shared/beegfs/piyush/datasets/AoTBench/AoTBench/data_files/{args.split}.json'
    data = su.io.load_json(data_file)
    data_name = args.split


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
        df_out.to_csv(f"outputs/aotbench_results-{model_name}-{data_name}.csv", index=False)
        print(f"Results saved to outputs/aotbench_results-{model_name}-{data_name}.csv")