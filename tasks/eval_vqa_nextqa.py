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


def process_row_tvbench(row, video_dir, n_frames=16):
    """
    This function is specialised for TVBench dataset.
    """
    video_path = f"{video_dir}/{row['video_file']}"
    assert os.path.exists(video_path), f"Video file not found: {video_path}"
    question = row['question']
    options = row['candidates']

    # Currently, these are hardcoded for TVBench dataset.
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
        'video_file': row['video_file'],
        'video': row['video'],
        'video_path': video_path,
        'n_frames': n_frames,
        'question': question,
        'options': options,
        'generated_answer': generated_answer,
        'indexed_options': indexed_options,
        'true_answer': row['answer'],
    }
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model_id",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint",
    )
    parser.add_argument("-d", "--dataset", type=str, default="nextqa-mc")
    args = parser.parse_args()
    
    # Load model
    model_id = args.model_id
    encoder = AutoEncoder.from_pretrained(model_id, device_map='auto')
    su.misc.num_params(encoder.model)
    
    if args.dataset == "nextqa-mc":
        # Load NextQA dataset
        data_dir = "/scratch/shared/beegfs/piyush/datasets/NExTQA"
        csv_path = f"{data_dir}/mc.csv"
        video_dir = f"{data_dir}/NExTVideo"
        assert os.path.exists(csv_path), f"CSV file not found: {csv_path}"
        df = pd.read_csv(csv_path)
        
        process_func = process_row_nextqa
        result_file = f"{su.log.repo_path}/results/nextqa_mc.npy"
    
    elif args.dataset == "tvbench":
        data_dir = "/scratch/shared/beegfs/piyush/datasets/TVBench"
        video_dir = f"{data_dir}/video"
        csv_path = f"{data_dir}/all_except_action_antonym.csv"
        assert os.path.exists(csv_path), f"CSV file not found: {csv_path}"
        df = pd.read_csv(csv_path)
        
        process_func = process_row_tvbench
        result_file = f"{su.log.repo_path}/results/tvbench.npy"
    else:
        raise ValueError(f"Dataset {args.dataset} not supported")
    
    # Result file (.npy to avoid JSON serialization issues with numpy types)
    # result_file = f"{su.log.repo_path}/results/nextqa_mc.npy"
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    
    # If result file exists, then drop those rows that are already in the result file
    if os.path.exists(result_file):
        print(f"Result file found: {result_file}")
        # Load existing results from .npy (list of dicts)
        try:
            existing_results = np.load(result_file, allow_pickle=True)
            if isinstance(existing_results, np.ndarray):
                existing_results = existing_results.tolist()
        except Exception as e:
            print(f"Failed to load existing results from npy: {e}")
            existing_results = []
        # Only need to update these results in the result file
        df = df[~df['video'].isin([result['video'] for result in existing_results])]
        print(f"Dropped {len(existing_results)} rows that are already in the result file")
        print(f"Number of rows left: {len(df)}")
        print("=" * 60)
    else:
        print(f"Result file not found: {result_file}")
        existing_results = []
        print("=" * 60)
    
    # Print number of rows to process
    print(f"Number of rows to process: {len(df)}")
    
    # Run on the entire dataset and save outputs as a JSON file (list of dicts)
    iterator = su.log.tqdm_iterator(range(len(df)), desc="Processing rows", total=len(df))
    save_freq = 20
    # Buffer for newly processed results; we will merge with existing_results on every save
    results_buffer = []
    for i in iterator:
        try:
            result = process_func(df.iloc[i], video_dir, n_frames=16)
            results_buffer.append(result)
            if (i+1) % save_freq == 0:
                print(f"Saving results with {len(results_buffer)} new rows (total will be {len(existing_results) + len(results_buffer)})")
                combined_results = existing_results + results_buffer
                # Save as numpy array of objects (list of dicts)
                np.save(result_file, np.array(combined_results, dtype=object))
                existing_results = combined_results
                results_buffer = []
        except Exception as e:
            print(f"Error processing row {i} of {len(df)}: {e}")
            print("Skipping this row...")
            continue
    print(f"Saving results with {len(results_buffer)} new rows (final total will be {len(existing_results) + len(results_buffer)})")
    combined_results = existing_results + results_buffer
    np.save(result_file, np.array(combined_results, dtype=object))