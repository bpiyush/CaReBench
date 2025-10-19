import os
import sys
import json
import argparse
import random

import torch
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from utils.video import read_frames_decord
from torchvision.transforms.v2 import PILToTensor
from models.modeling_encoders import AutoEncoder

import shared.utils as su
from models.modeling_basemodels import EOL_PROMPTS
from qwen_vl_utils import process_vision_info


def recall_at_k(scores, positive_pairs, k):
    """
    Compute the recall at k for each sample
    :param scores: compability score between  text and image embeddings (nb texts, nb images)
    :param k: number of images to consider per text, for retrieval
    :param positive_pairs: boolean matrix of positive pairs (nb texts, nb images)
    :return: recall at k averaged over all texts
    """
    nb_texts, nb_images = scores.shape
    # for each text, sort according to image scores in decreasing order
    topk_indices = torch.topk(scores, k, dim=1)[1]
    # compute number of positives for each text
    nb_positive = positive_pairs.sum(dim=1)
    # nb_texts, k, nb_images
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    # compute number of true positives
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    # a true positive means a positive among the topk
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1,2))
    # compute recall at k
    recall_at_k = (nb_true_positive / nb_positive)
    return recall_at_k


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def compute_retrieval_metrics(scores, positive_pairs, device, recall_k_list=[1, 5, 10], batch_size=64):
    """
    Compute retrieval metrics (recall@k for both image and text retrieval)
    """
    metrics = {}
    for recall_k in recall_k_list:
        # Image retrieval: for each text, find the best matching videos
        metrics[f"image_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores, positive_pairs, batch_size, device, k=recall_k)>0).float().mean().item() * 100
        # Text retrieval: for each video, find the best matching texts
        metrics[f"text_retrieval_recall@{recall_k}"] = (batchify(recall_at_k, scores.T, positive_pairs.T, batch_size, device, k=recall_k)>0).float().mean().item() * 100
    
    return metrics


def get_video_embedding(duration, video_path, encoder):
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                'fps': 32. / duration,
                "resized_height": 256,
                "resized_width": 455,
            },
            {"type": "text", "text": EOL_PROMPTS['video']},
        ],
    }]
    prompt = encoder.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

    inputs = encoder.processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        # **video_kwargs,
    )
    inputs = inputs.to("cuda")
    with torch.inference_mode():
        output = encoder.model.generate(
            **inputs,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        z = output.hidden_states[0][-1][:, -1, :].cpu().float()
    return z


def main():
    parser = argparse.ArgumentParser(description='Evaluate retrieval performance')
    parser.add_argument(
        '--model_path',
        type=str,
        default="/work/piyush/experiments/CaRe/special_milestones/care-stage2-nli-27k-ego4d-3k",
        help='Path to the model checkpoint',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        default=False,
        help='Run in debug mode with 100 random samples',
    )
    parser.add_argument(
        '--data',
        type=str,
        default="carebench",
        help='Data to evaluate on',
    )
    
    args = parser.parse_args()
    
    num_frames = 32
    trim30 = False

    # Load model
    encoder = AutoEncoder.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        device_map='cuda:0',
    )
    
    # Load data
    data = args.data
    data_path = "./data.json"
    assert os.path.exists(data_path), f"{data_path} not found"
    with open(data_path) as f:
        data_configs = json.load(f)
    data_config = data_configs[data]
    anno_path = data_config['anno_path']
    with open(anno_path) as f:
        data = json.load(f)
    
    # Debug mode: select 100 random samples
    if args.debug:
        data = random.sample(data, min(100, len(data)))
        
    
    # Compute embeddings
    video_embs = []
    text_embs = []
    iterator = su.log.tqdm_iterator(data, desc='Computing embeddings')
    for item in iterator:
        video_path = f"{data_config['data_root']}/{item['video']}"
        assert os.path.exists(video_path)
        caption = item['caption']
        
        with torch.no_grad():
            zt = encoder.encode_text(caption).cpu().float()
        
        duration = su.video.get_duration(video_path)
        zv = get_video_embedding(duration, video_path, encoder)
        video_embs.append(zv)
        text_embs.append(zt)
        
    video_embs = torch.cat(video_embs)
    text_embs = torch.cat(text_embs)
    
    # Compute scores
    scores = text_embs @ video_embs.t()
    
    # Normalize embeddings
    text_embs = F.normalize(text_embs, dim=-1)
    video_embs = F.normalize(video_embs, dim=-1)
    
    # Recompute scores with normalized embeddings
    scores = text_embs @ video_embs.t()
    
    # Check for NaN values
    assert text_embs.isnan().sum().item() == 0, 'nan in text embeddings'
    assert video_embs.isnan().sum().item() == 0, 'nan in video embeddings'
    
    # For 1:1 text-video pairs, positive pairs are on the diagonal
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), torch.arange(len(scores))] = True
    
    # Compute metrics
    metrics = compute_retrieval_metrics(scores, positive_pairs, device="cuda")
    
    # Print metrics
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")


if __name__ == "__main__":
    main()