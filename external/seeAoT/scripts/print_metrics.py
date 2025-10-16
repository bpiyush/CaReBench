"""Print metrics after the evaluation script is run."""

import json
import pandas as pd
import shared.utils as su
import glob
from termcolor import colored

    
if __name__ == "__main__":
    repo_path = su.log.repo_path
    data_dir = f"{repo_path}/external/seeAoT/data"
    
    datasets = ["UCF101", "Rtime_t2v", "ReverseFilm"]
    for dataset in datasets:
        print(colored(f"Processing {dataset}...", "yellow"))
        
        inputs_dir = f"{data_dir}/data_files/input"
        output_dir = f"{data_dir}/data_files/output"
        
        input_file = f"{inputs_dir}/{dataset}.json"
        data_input = su.io.load_json(input_file)
        data_input = pd.DataFrame(data_input)
        
        output_files = glob.glob(f"{output_dir}/{dataset}/*.jsonl")
        for output_file in output_files:
            print(output_file)
            data_output = su.io.load_jsonl(output_file)
            data_output = pd.DataFrame(data_output)
            print("Number of samples: ", len(data_output))
            
            if len(data_output) != len(data_input):
                print(colored(f"Number of samples mismatch: {len(data_output)} != {len(data_input)}", "red"))
                continue
            
            # Compute accuracy
            data_output['qa_idx'] = data_output['idx']
            df = pd.merge(data_input, data_output, on='qa_idx', how='left')
            
            accuracy = (df['response'].apply(lambda x: x.strip(".")) == df['ans']).mean() * 100.
            print(f"Accuracy: {accuracy:.2f}%")

            print("=" * 60)
        
        print(colored("-" * 60, "yellow"))