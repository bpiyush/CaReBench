"""Generates and saves results for VQA on NextQA-MC subset."""
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
from models.modeling_encoders import AutoEncoder
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)


def convert_to_prompt(messages):
    """
    Convert a list of message dictionaries to a prompt string.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' fields
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    for message in messages:
        role = message["role"].upper()
        prompt += f"{role}: "
        
        content_items = message["content"]
        for item in content_items:
            if item["type"] == "video":
                prompt += "<video>\n"
            elif item["type"] == "text":
                prompt += item["text"]
        
        prompt += " "
    
    prompt += "ASSISTANT: "
    
    return prompt


def generate_answer_for_videoqa(encoder, video_path, question, options, n_frames=16, generate_kwargs={}):
    """
    Generates an answer for VideoQA.

    Args:
        video_path (str): video path
        question (str): question
        options (list[str]): list out all the options
        n_frames (int): number of frames
        generate_kwargs (dict): additional kwargs for model.generate
    """
    assert os.path.exists(video_path)

    # Convert options into a single string
    indexed_options = [f"{j}: {v}" for j, v in enumerate(options)]
    option_string = '\n'.join(indexed_options)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": 8.0,
                },
                {
                    "type": "text",
                    "text": f"""Answer the following question by choosing the right option from provided choices. \n
                    Question: {question} \n
                    Options: \n {option_string}
                    """
                },
            ],
        }
    ]

    # Convert into a single string prompt
    prompt = convert_to_prompt(messages)

    # Prepare video
    pixel_values = read_frames_decord(video_path, n_frames).unsqueeze(0)
    pixel_values = transform_pixel_values(pixel_values)
    nframes = pixel_values.shape[1]
    to_image = ToPILImage()
    batched_frames = []
    for batch in pixel_values:
        frames = [to_image(v) for v in batch]
        batched_frames.append(frames)

    # Run through model
    for frames in batched_frames:
        input_prompt = prompt.replace("<video>", "<image>"*len(frames))
        input_ids = encoder.processor.get_text_inputs(input_prompt)
        frames = encoder.processor.get_pixel_values(frames)
        inputs = {
            "input_ids": input_ids,
            "pixel_values": frames,
        }
        inputs = {k:v.to(encoder.model.device) for k,v in inputs.items() if v is not None}
        outputs = encoder.model.generate(
            **inputs,
            **generate_kwargs,
        )
        output_text = encoder.processor.tokenizer.decode(
            outputs[0][inputs['input_ids'][0].shape[0]:], skip_special_tokens=True,
        )
        break # Safe to break since it is only a single sample
    return output_text, indexed_options


def process_row_nextqa(row, video_dir, n_frames=16):
    """
    This function is specialised for NEXTQA-MC dataset.
    """
    video_path = f"{video_dir}/{row['video']}.mp4"
    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    # Question
    question = row['question']
    if not question.endswith("?"):
        question += '?'

    # Prepare options for MCQ
    options = [
        row['a0'], row['a1'], row['a2'], row['a3'], row['a4'],
    ]
    
    # Currently, these are hardcoded for NEXTQA-MC dataset.
    generate_kwargs = {
        "do_sample": False,
        "max_new_tokens": 128,
        "top_p": 1,
        "temperature": 0.,
        "use_cache": True,
    }

    generated_answer, indexed_options = generate_answer_for_videoqa(
        encoder=encoder, 
        video_path=video_path,
        question=question,
        options=options,
        n_frames=n_frames,
        generate_kwargs=generate_kwargs,
    )
    
    # Return result item as dict
    result = {
        'video': row['video'],
        'video_path': video_path,
        'n_frames': n_frames,
        'question': question,
        'options': options,
        'generated_answer': generated_answer,
        'indexed_options': indexed_options,
        'true_answer': indexed_options[row['answer']],
    }
    return result


if __name__ == "__main__":
    
    # Load model
    model_id = "/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint"
    encoder = AutoEncoder.from_pretrained(model_id, device_map='auto')
    su.misc.num_params(encoder.model)
    
    # Load NextQA dataset
    data_dir = "/scratch/shared/beegfs/piyush/datasets/NExTQA"
    csv_path = f"{data_dir}/mc.csv"
    video_dir = f"{data_dir}/NExTVideo"
    assert os.path.exists(csv_path), f"CSV file not found: {csv_path}"
    df = pd.read_csv(csv_path)
    
    # Result file
    result_file = f"{su.log.repo_path}/results/nextqa_mc.json"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    # If result file exists, then drop those rows that are already in the result file
    if os.path.exists(result_file):
        print(f"Result file found: {result_file}")
        results = su.io.load_json(result_file)
        # Only need to update these results in the result file
        df = df[~df['video'].isin([result['video'] for result in results])]
        print(f"Dropped {len(results)} rows that are already in the result file")
        print(f"Number of rows left: {len(df)}")
        print("=" * 60)
    else:
        print(f"Result file not found: {result_file}")
        results = []
        print("=" * 60)
    
    # Print number of rows to process
    print(f"Number of rows to process: {len(df)}")
    
    # Run on the entire dataset and save outputs as a JSON file (list of dicts)
    iterator = su.log.tqdm_iterator(range(len(df)), desc="Processing rows", total=len(df))
    save_freq = 20
    for i in iterator:
        try:
            result = process_row_nextqa(df.iloc[i], video_dir, n_frames=16)
            results.append(result)
            if (i+1) % save_freq == 0:
                print(f"Saving results with {len(results)} rows")
                su.io.save_json(results, result_file)
                results = []
        except Exception as e:
            print(f"Error processing row {i} of {len(df)}: {e}")
            print("Skipping this row...")
            continue
    print(f"Saving results with {len(results)} rows")
    su.io.save_json(results, result_file)