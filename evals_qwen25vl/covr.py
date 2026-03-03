import os
import sys
import json
import torch
import numpy as np
import pandas as pd

import shared.utils as su
from transformers import Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration
from models.qwen25vl import EncoderForQwen25VL


model_path = "/work/piyush/pretrained_checkpoints/ArrowRL-Qwen2.5-VL-7B"
model = EncoderForQwen25VL.from_pretrained(model_path, torch_dtype=torch.float16, attn_implementation='flash_attention_2', device_map='auto')
su.misc.num_params(model.model)


def load_data(dataset: str = 'covr') -> pd.DataFrame:
    """
    Load dataset configuration and CSV file for CoVR.
    
    Args:
        dataset: Name of dataset to load (covr)
        
    Returns:
        DataFrame with video paths and metadata
    """
    import json
    
    # Load dataset config from YAML
    cfg_path = "datasets.yaml"
    assert os.path.exists(cfg_path), f"Dataset config file {cfg_path} does not exist"
    all_configs = su.io.load_yml(cfg_path)
    
    # Validate dataset
    assert dataset in all_configs, f"Dataset {dataset} not found in datasets.yaml"
    
    data_config = all_configs[dataset]
    
    # Load CSV
    csv_path = data_config['csv_path']
    video_dir = data_config['video_dir']
    
    assert os.path.exists(csv_path), f"CSV file {csv_path} does not exist"
    df = pd.read_csv(csv_path)
    
    print(f"Dataset: {dataset}")
    print(f"Number of rows in CoVR-test: {len(df)}")
    
    # Add video paths
    df['video1_path'] = df['video1'].apply(lambda x: f"{video_dir}/{x}")
    df['video2_path'] = df['video2'].apply(lambda x: f"{video_dir}/{x}")
    
    # Filter to only existing videos
    df = df[df.video1_path.apply(os.path.exists) & df.video2_path.apply(os.path.exists)]
    print(f"Number of rows with all videos available: {df.shape}")
    
    # Remove problematic videos (if any)
    problematic_videos = data_config.get('problematic_videos', [])
    for video_path in problematic_videos:
        df = df[df.video1_path != video_path]
    
    print(f"Sample row: ")
    print(json.dumps(df.iloc[0].to_dict(), indent=4))

    return df


from typing import Dict
def gather_metrics(
    query_embeds: Dict[str, torch.Tensor],
    candidates: Dict[str, torch.Tensor],
    df: pd.DataFrame,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Compute retrieval metrics for CoVR.
    
    Args:
        query_embeds: Dictionary of query embeddings
        candidates: Dictionary of candidate video embeddings
        df: DataFrame with ground truth mappings
        verbose: Whether to print progress
        
    Returns:
        Dictionary of retrieval metrics
    """
    from utils.general_retrieval_metrics import itm_eval
    
    zq = []
    zc = []
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        query_key = f"{row['edit']}|{row['video1']}"
        candi_key = row['video2']
        if query_key not in query_embeds or candi_key not in candidates:
            if verbose:
                print(f"Missing value for {i}. Skipped.")
            continue
        zq.append(query_embeds[query_key])
        zc.append(candidates[candi_key])
    
    zq = torch.stack(zq).numpy()
    zc = torch.stack(zc).numpy()
    
    if verbose:
        print(f"Query embeddings shape: {zq.shape}")
        print(f"Candidate embeddings shape: {zc.shape}")
    
    # i:q and t:c; and we care about q2c metrics, i.e., i2t, i.e., text_*
    score_q2c = zq @ zc.T
    score_c2q = zc @ zq.T
    indices = {i: i for i in range(len(score_q2c))}
    metrics = itm_eval(
        scores_i2t=score_q2c,
        scores_t2i=score_c2q,
        txt2img=indices,
        img2txt=indices,
        verbose=verbose
    )
    return metrics


df = load_data(dataset="covr")
# Load dataset config
cfg_path = "datasets.yaml"
all_configs = su.io.load_yml(cfg_path)
data_config = all_configs["covr"]
video_dir = data_config['video_dir']
print('-' * 100)


# Compute video embeddings for the candidate videos
videos = set(df.video2.tolist())
candidates = {}
for video in su.log.tqdm_iterator(videos, desc='Computing features for candidate videos'):
    video_path = f"{video_dir}/{video}"
    assert os.path.exists(video_path)
    with torch.no_grad():
        zv = model.encode_vision(video_path).squeeze(0)
        zv = torch.nn.functional.normalize(zv, dim=-1)
        zv = zv.cpu().float()
    candidates[video] = zv
print(f"Successfully computed {len(candidates)} candidate embeddings.")


# Gather query embeddings
query_embeds = {}
for i in su.log.tqdm_iterator(range(len(query_embeds), len(df)), desc="Compute query embeddings"):
    row = df.iloc[i].to_dict()
    video_path = f"{video_dir}/{row['video1']}"
    edit_text = row['edit']
    with torch.no_grad():
        try:
            zv = model.encode_vision_text(video_path, edit_text).squeeze(0)
            zv = torch.nn.functional.normalize(zv, dim=-1)
            zv = zv.cpu().float()
            key = f"{edit_text}|{row['video1']}"
            query_embeds[key] = zv
        except:
            print(f"Skpping {i}")
            continue
print(f"Successfully computed {len(query_embeds)} query embeddings.")


# Compute retrieval metrics
print("Computing retrieval metrics...")
metrics = gather_metrics(query_embeds, candidates, df, verbose=True)

print('-' * 100)
print("Final Results:")
print(json.dumps(metrics, indent=4))
print('-' * 100)

# Save results to file
# result_dir = "./results"
result_dir = f"{model_path}/metrics"
os.makedirs(result_dir, exist_ok=True)
dataset = "covr"
result_path = f"{result_dir}/metrics_covr_{dataset}.json"

# Add metadata to results
metrics['dataset'] = dataset
metrics['model_path'] = model_path
# metrics['num_frames'] = args.num_frames
metrics['debug'] = False

with open(result_path, 'w') as f:
    json.dump(metrics, f, indent=4)

print(f"Results saved to {result_path}")
