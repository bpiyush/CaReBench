import os
import sys
import argparse

import pandas as pd
import numpy as np

import shared.utils as su
from utils.qwen3_utils import QwenWrapper


def parse_args():
    parser = argparse.ArgumentParser(description="Modify edit instructions using Qwen model")
    parser.add_argument("--si", type=int, default=None, help="Start index for subset selection")
    parser.add_argument("--ei", type=int, default=None, help="End index for subset selection")
    parser.add_argument("--attn", type=str, default='flash_attention_2', help="Attention implementation")
    return parser.parse_args()


def create_prompt(txt1, txt2, edit):
    sent = f"Source caption: {txt1}\nEdit instruction: {edit}\nTarget caption: {txt2}\n\n"
    task = """
            When the source caption is edited with the given instruction,
            it results in the given target caption.
            
            Reformulate the edit instruction to make it clearer.
            If you think it is already clear, then output the same.
            """
    prompt = f"{task}\n\n{sent}"
    return prompt


if __name__ == "__main__":
    args = parse_args()
    
    anno_dir = "/scratch/shared/beegfs/piyush/datasets/WebVid-CoVR/annotations"
    df = pd.read_csv(f"{anno_dir}/text_triplets-v1.csv")
    
    # Select subset based on start and end indices
    si = args.si if args.si is not None else 0
    ei = args.ei if args.ei is not None else len(df)
    df = df.iloc[si:ei].reset_index(drop=True)
    print(f"Processing rows from index {si} to {ei} (total: {len(df)} rows)")
    
    model_path = "/work/piyush/pretrained_checkpoints/Qwen3-4B-Instruct-2507/"
    model = QwenWrapper(model_name=model_path, attn_implementation=args.attn)
    
    iterator = su.log.tqdm_iterator(range(len(df)), desc="Processing rows")
    df_results = []
    for i in iterator:
        row = df.iloc[i].to_dict()
        
        try:
            prompt = create_prompt(row['txt1'], row['txt2'], row['edit'])
            result = model.generate_answer(prompt, max_new_tokens=128)
            df_results.append(
                {**row, "edit_modified_result": result['content']}
            )
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            continue
    
    # Save the results
    df_results = pd.DataFrame(df_results)
    print("Number of rows processed:", len(df_results))
    
    # Build output filename with index range if specified
    if args.si is not None or args.ei is not None:
        output_filename = f"text_triplets-v1_edit_modified_si{si}_ei{ei}.csv"
    else:
        output_filename = "text_triplets-v1_edit_modified.csv"
    df_results.to_csv(f"{anno_dir}/{output_filename}", index=False)