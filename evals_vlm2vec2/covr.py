import os
import sys
import json
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import decord
from torchvision.transforms.v2 import ToPILImage

from shared.utils.log import tqdm_iterator
from shared.utils.io import load_yml
# from utils.video import read_frames_decord
# from utils.model import transform_pixel_values
# from models.modeling_encoders import AutoEncoder


def load_data(dataset: str = 'covr') -> pd.DataFrame:
    """
    Load dataset configuration and CSV file for CoVR.
    
    Args:
        dataset: Name of dataset to load (covr)
        
    Returns:
        DataFrame with video paths and metadata
    """
    # Load dataset config from YAML
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets.yaml")
    assert os.path.exists(cfg_path), f"Dataset config file {cfg_path} does not exist"
    all_configs = load_yml(cfg_path)
    
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


PROMPT = "<video>\nEdit instruction: <sent>\n"

def embed_video_text(video_path, edit_text):
    prompt = PROMPT.replace("<sent>", edit_text)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Imagine the given text edit instruction applied on the given video. Represent the resulting video in one word.',
        videos=video_inputs,
        return_tensors="pt"
    )
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
    inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
    qry_output = model(qry=inputs)["qry_reps"]
    return qry_output.squeeze(0).float()


def embed_video(video_path):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": "Describe this video."},
            ],
        }
    ]

    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=f'{VLM_VIDEO_TOKENS[QWEN2_VL]} Represent the given video.',
        videos=video_inputs,
        return_tensors="pt"
    )
    inputs = {key: value.to('cuda') for key, value in inputs.items()}
    inputs['pixel_values_videos'] = inputs['pixel_values_videos'].unsqueeze(0)
    inputs['video_grid_thw'] = inputs['video_grid_thw'].unsqueeze(0)
    tgt_output = model(tgt=inputs)["tgt_reps"]
    return tgt_output.squeeze(0).float()


if __name__ == "__main__":
    model_path = "/work/piyush/pretrained_checkpoints/VLM2Vec-V2.0"
    
    df = load_data(dataset="covr")
    # Load dataset config
    cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datasets.yaml")
    all_configs = load_yml(cfg_path)
    data_config = all_configs["covr"]
    video_dir = data_config['video_dir']
    print('-' * 100)

    
    repo_dir = "/users/piyush/projects/VLM2Vec"
    sys.path.append(repo_dir)
    
    from src.arguments import ModelArguments, DataArguments
    from src.model.model import MMEBModel
    from src.model.processor import load_processor, QWEN2_VL, VLM_VIDEO_TOKENS
    import torch
    from src.model.vlm_backbone.qwen2_vl.qwen_vl_utils import process_vision_info


    model_args = ModelArguments(
        model_name='/work/piyush/pretrained_checkpoints/VLM2Vec-V2.0/',
        pooling='last',
        normalize=True,
        model_backbone='qwen2_vl',
        lora=True
    )
    data_args = DataArguments()
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args)
    model = model.to('cuda', dtype=torch.bfloat16)
    model.eval()

    # Compute video embeddings for the candidate videos
    videos = set(df.video2.tolist())
    candidates = {}
    for video in tqdm_iterator(videos, desc='Computing features for candidate videos'):
        video_path = f"{video_dir}/{video}"
        assert os.path.exists(video_path)
        with torch.no_grad():
            zv = embed_video(video_path)
            zv = torch.nn.functional.normalize(zv, dim=-1)
            zv = zv.cpu().float()
        candidates[video] = zv
    print(f"Successfully computed {len(candidates)} candidate embeddings.")

    # Gather query embeddings
    query_embeds = {}
    from shared.utils.log import tqdm_iterator
    for i in tqdm_iterator(range(len(query_embeds), len(df)), desc="Compute query embeddings"):
        row = df.iloc[i].to_dict()
        video_path = f"{video_dir}/{row['video1']}"
        edit_text = row['edit']
        with torch.no_grad():
            try:
                zv = embed_video_text(video_path, edit_text)
                zv = torch.nn.functional.normalize(zv, dim=-1)
                zv = zv.cpu().float()
                key = f"{edit_text}|{row['video1']}"
                query_embeds[key] = zv
            except:
                print(f"Skpping {i}")
                continue
    print(f"Successfully computed {len(query_embeds)} query embeddings.")
    
    import ipdb; ipdb.set_trace()
    
    # Compute retrieval metrics
    metrics = gather_metrics(query_embeds, candidates, df)
    
    # Save metrics
    result_dir = f"{model_path}/metrics"
    model_name = "vlm2vec2"
    os.makedirs(result_dir, exist_ok=True)
    with open(os.path.join(result_dir, f"metrics_covr_{model_name}.json"), "w") as f:
        json.dump(metrics, f, indent=4)