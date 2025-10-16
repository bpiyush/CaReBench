"""Print metrics after the evaluation script is run."""

import json
import pandas as pd
import shared.utils as su

    
if __name__ == "__main__":
    repo_path = su.log.repo_path
    data_dir = f"{repo_path}/external/seeAoT/data"
    
    dataset = "UCF101"
    
    inputs_dir = f"{data_dir}/data_files/input"
    output_dir = f"{data_dir}/data_files/output"
    
    input_file = f"{inputs_dir}/{dataset}.json"
    data_input = su.io.load_json(input_file)
    output_file = f"{output_dir}/{dataset}/16frames_.jsonl"
    data_output = su.io.load_jsonl(output_file)
    print("Number of samples: ", len(data_input))
    
    df_input = pd.DataFrame(data_input)
    df_output = pd.DataFrame(data_output)
    df_output['qa_idx'] = df_output['idx']
    df = pd.merge(df_input, df_output, on='qa_idx', how='left')
    
    accuracy = (df.response.apply(lambda x: x.strip(".")) == df.ans).mean() * 100.
    print(f"Accuracy: {accuracy:.2f}%")
