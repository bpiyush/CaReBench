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

import shared.utils as su
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)


# Load model
from models.modeling_encoders import AutoEncoder
model_id = "/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/"\
    "nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint"
encoder = AutoEncoder.from_pretrained(model_id, device_map='auto')
su.misc.num_params(encoder.model)


def load_jsonl_data(file_path):
    """Load JSONL data from file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

data_dir = "/scratch/shared/beegfs/piyush/datasets/CameraBench"
jsonl_file = f"{data_dir}/t2v_metrics/camerabench/data/binary_classification/Move_Down.jsonl"
assert os.path.exists(jsonl_file)
data = load_jsonl_data(jsonl_file)


video_base_path = f"{data_dir}/videos"
question_template="{} Please only answer Yes or No."
answer_template="Yes"
results = []
# Process one sample at a time
for item in su.log.tqdm_iterator(data, desc="Computing VQA scores"):
    video_path = item['image']  # Note: using 'image' key for video path
    question = item['question']
    label = item['label']
    
    # Create result entry with metadata
    result_entry = {
        "video_path": video_path,
        "question": question,
        "ground_truth_label": label,
        # "method": f"{model_name}" + (f"_{checkpoint_name}" if checkpoint_name else ""),
        "score": None,
        "error": None
    }
    
    # Construct full video path
    full_video_path = os.path.join(video_base_path, video_path)

    question = question_template.format(question)
    answer = answer_template.format(question)
    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": 1,
        "top_p": 1.0,
        "temperature": 0,
        "use_cache": True,
        "output_scores": True,
        "return_dict_in_generate": True,
    }

    with torch.no_grad():

        pixel_values = read_frames_decord(full_video_path, num_frames=16).unsqueeze(0)
        pixel_values = transform_pixel_values(pixel_values) # [B, T, C, H, W]
        nframes = pixel_values.shape[1]
        # prompt = encoder.image_eol_prompt if nframes == 1 else encoder.video_eol_prompt
        prompt = f"USER:<video>\n{question}\n\nASSISTANT:"
        to_image = ToPILImage()
        batched_frames = []
        for batch in pixel_values:
            frames = [to_image(v) for v in batch]
            batched_frames.append(frames)
        
            for frames in batched_frames:
                input_prompt = prompt.replace("<video>", "<image>"*len(frames))
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
            # output_text = encoder.processor.tokenizer.decode(
            #     outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True,
            # )
            scores = outputs.scores[0]
            probs = torch.nn.functional.softmax(scores, dim=-1)
            yes_token_id = encoder.processor.tokenizer.encode(answer)[-1]
            lm_prob = probs[0, yes_token_id].item()

            # # The other option is to only pick softmax over the given choices
            # yes_token_id = encoder.processor.tokenizer.encode('Yes')[-1]
            # no_token_id = encoder.processor.tokenizer.encode('No')[-1]
            # options_ids = [yes_token_id, no_token_id]
            # probs_binary = torch.nn.functional.softmax(scores[0][options_ids], dim=-1)
            # # ans_token_id = encoder.processor.tokenizer.encode(answer)[-1]
            # # lm_prob = probs_binary[options_ids.index(ans_token_id)].cpu().item()
            # lm_prob = probs_binary[0].cpu().item()

    
            # import ipdb; ipdb.set_trace()
    
            break # safe to break, since only a single video in a batch
    
    result_entry["score"] = float(lm_prob)
    results.append(result_entry)
import ipdb; ipdb.set_trace()

scores = np.array([r['score'] for r in results])
labels = np.array([1 if r["ground_truth_label"].lower() == 'yes' else 0 for r in results])
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(labels, scores)
print(f"Average Precision: {average_precision}")
import ipdb; ipdb.set_trace()