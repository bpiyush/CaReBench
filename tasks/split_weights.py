import os
from models.modeling_basemodels import AutoBase
import torch

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', "--model_name_or_path", type=str, default="/work/piyush/pretrained_checkpoints/CaRe-7B-Stage-1")
args = parser.parse_args()

model_name_or_path = args.model_name_or_path

# Remove the dest_dir if it exists
dest_dir = f"{model_name_or_path}-llm"
if os.path.isdir(dest_dir):
    os.rmdir(dest_dir)

base_model = AutoBase.from_pretrained(
    model_name_or_path,
    load_llm=True,
    device_map='cuda',
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

# Do a du -sh on the dest_dir
print(f"Size of {dest_dir}: {os.popen(f'du -sh {dest_dir}/*').read().strip()}")