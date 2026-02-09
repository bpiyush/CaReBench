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

from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict as edict
import numpy as np
import json
import einops

import shared.utils as su
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)
from models.modeling_encoders import AutoEncoder
from notebooks.eval_care_retrieval import load_model, load_data
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
    functional,
)
from utils.video import read_frames_decord
from models.modeling_encoders import AutoEncoder


def extract_topk_tokens(model, frames_raw, k=25):
    with torch.no_grad():
        zv = model.encode_vision(frames_raw.unsqueeze(0))
        logits = model.model.language_model.lm_head(zv).cpu().float().squeeze(0)

    # Pick up top-K tokens to look at and their logit values
    topk = torch.topk(logits, k=k, dim=-1)
    topk_values = topk.values
    topk_indices = topk.indices

    # Decode into strings
    token_strings = [model.processor.tokenizer.decode([t], skip_special_tokens=False) for t in topk_indices]

    return token_strings, topk_values


def plot_logits(token_strings, values, ax, cmap=None, label_fontsize=10):
    """
    Plot vertical bars for token logits, place token labels inside bars,
    and color bars according to logit magnitude (darker = larger).
    
    Args:
        token_strings: list of str, labels for each bar (same order as values)
        values: array-like or torch.Tensor of shape (K,)
        ax: matplotlib Axes to draw on
        cmap: matplotlib colormap (optional). Defaults to plt.cm.Blues.
        label_fontsize: int font size for labels placed inside bars
    Returns:
        bars: BarContainer from ax.bar
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if cmap is None:
        cmap = plt.cm.Blues

    # --- Normalize values, accepting torch.Tensor or numpy/list ---
    try:
        # if torch is available and it's a tensor, convert safely
        import torch
        if isinstance(values, torch.Tensor):
            vals = values.detach().cpu().numpy()
        else:
            vals = np.asarray(values)
    except Exception:
        vals = np.asarray(values)

    vals = np.array(vals, dtype=float).ravel()
    if vals.size != len(token_strings):
        raise ValueError("length of token_strings must match length of values")

    # --- Color mapping (higher -> darker) ---
    vmin = float(vals.min())
    vmax = float(vals.max())
    denom = (vmax - vmin) if (vmax != vmin) else 1.0
    norm = (vals - vmin) / denom  # in [0,1]
    # map norm to colormap range (avoid the very pale end)
    color_values = [cmap(0.25 + 0.75 * n) for n in norm]

    # --- Draw bars ---
    x = np.arange(len(token_strings))
    bars = ax.bar(x, vals, color=color_values, edgecolor='none')

    # axes labels + grid
    ax.set_xlabel("Token")
    ax.set_ylabel("Logit Value")
    ax.grid(axis="y", alpha=0.4)

    # remove crowded x-ticks; we'll draw labels inside bars instead
    ax.set_xticks([])

    # --- Put token labels inside each bar ---
    for i, (bar, tok, n) in enumerate(zip(bars, token_strings, norm)):
        height = bar.get_height()
        # Choose readable text color depending on darkness (norm)
        text_color = "white" if n > 0.55 else "black"
        # place text centered horizontally, at ~50% of the bar height
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            max(0.02 * (vmax - vmin), height * 0.5),  # small floor so very small bars still show label
            tok,
            ha="center",
            va="center",
            rotation=90,            # rotate to match your example
            fontsize=label_fontsize,
            color=text_color,
            clip_on=True,
        )

    # tighten layout so labels aren't clipped
    ax.figure.tight_layout()
    return bars


import numpy as np
import torch
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

class SemanticMRREvaluator:
    def __init__(self, model_name: str = "/work/piyush/pretrained_checkpoints/all-MiniLM-L6-v2"):
        """
        Initialize with a lightweight sentence embedding model.
        
        Args:
            model_name: HuggingFace model name. Options:
                - "sentence-transformers/all-MiniLM-L6-v2" (lightweight, 80MB)
                - "sentence-transformers/all-mpnet-base-v2" (better quality, 420MB)
        """
        print(f"Loading embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        
        # Use GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a single text."""
        # Tokenize
        encoded = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**encoded)
            # Mean pooling
            embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()[0]
    
    def mean_pooling(self, model_output, attention_mask):
        """Mean pooling - take average of all token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity between two texts."""
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        return np.dot(emb1, emb2)  # Already normalized, so dot product = cosine similarity
    
    def compute_mrr(self, 
                    top_k_tokens: List[str], 
                    ground_truth_verb: str,
                    similarity_threshold: float = 0.5) -> Tuple[float, List[Tuple[int, str, float]]]:
        """
        Compute MRR using semantic similarity.
        
        Args:
            top_k_tokens: List of K tokens sorted by logits (highest first)
            ground_truth_verb: The ground truth verb to match against
            similarity_threshold: Minimum cosine similarity to consider a match (0-1)
            
        Returns:
            Tuple of (mrr_score, matches) where matches is list of (rank, token, similarity)
        """
        # Get embedding for ground truth verb
        gt_embedding = self.get_embedding(ground_truth_verb)
        
        matches = []
        first_match_rank = None
        
        # Check each token in order
        for rank, token in enumerate(top_k_tokens, start=1):
            # Get token embedding and compute similarity
            token_embedding = self.get_embedding(token)
            similarity = np.dot(gt_embedding, token_embedding)
            
            # Record if it's above threshold
            if similarity >= similarity_threshold:
                matches.append((rank, token, float(similarity)))
                if first_match_rank is None:
                    first_match_rank = rank
        
        # Compute MRR
        mrr = 1.0 / first_match_rank if first_match_rank else 0.0
        
        return mrr, matches
    
    def compute_mrr_batch(self,
                         top_k_tokens: List[str],
                         ground_truth_verb: str,
                         similarity_threshold: float = 0.5) -> Tuple[float, List[Tuple[int, str, float]]]:
        """
        Optimized version: compute all similarities in one batch.
        Much faster for large K.
        """
        if len(top_k_tokens) == 0:
            return 0.0, []
        
        # Get all embeddings in batch
        gt_embedding = self.get_embedding(ground_truth_verb)
        
        # Batch encode all tokens
        encoded = self.tokenizer(
            top_k_tokens,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**encoded)
            embeddings = self.mean_pooling(outputs, encoded['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # Compute similarities
        gt_tensor = torch.tensor(gt_embedding).to(self.device)
        similarities = torch.matmul(embeddings, gt_tensor).cpu().numpy()
        
        # Find matches
        matches = []
        first_match_rank = None
        
        for rank, (token, similarity) in enumerate(zip(top_k_tokens, similarities), start=1):
            if similarity >= similarity_threshold:
                matches.append((rank, token, float(similarity)))
                if first_match_rank is None:
                    first_match_rank = rank
        
        mrr = 1.0 / first_match_rank if first_match_rank else 0.0
        
        return mrr, matches


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/work/piyush/pretrained_checkpoints/TARA')
    args = parser.parse_args()
    
    device_map = "auto"
    attn_implementation = 'flash_attention_2'

    model_path = args.model_path
    model = AutoEncoder.from_pretrained(
        model_path,
        device_map=device_map,
        attn_implementation=attn_implementation,
        use_flash_attn=False,
        dtype=torch.bfloat16,
    )
    su.misc.num_params(model.model)
    

    # Load data

    dataset = 'ssv2'
    label_col = 'template'

    # dataset = 'epic'
    # label_col = 'narration'

    df = load_data(dataset)
    df = df.drop_duplicates(subset=['id', 'text_id']).reset_index(drop=True)

    # Chiral IDs
    chiral_triplet_ids = df.chiral_triplet_id.unique()
    
    evaluator = SemanticMRREvaluator()
    
    iterator = su.log.tqdm_iterator(range(len(df)), desc='Computing logit rankings')
    outputs = {'mrr': [], 'matches': []}
    for i in iterator:
        row = df.iloc[i].to_dict()
        video_path = row['video_path']
        frames_raw = read_frames_decord(video_path, 8)
        tokens, logits = extract_topk_tokens(model, frames_raw)
        mrr, matches = evaluator.compute_mrr_batch(
            top_k_tokens=tokens,
            ground_truth_verb=row[label_col],
            similarity_threshold=0.4
        )
        outputs['mrr'].append(mrr)
        outputs['matches'].append(matches)
    avg_mrr = np.mean(outputs['mrr'])
    print(f"Average MRR: {avg_mrr}")
    import ipdb; ipdb.set_trace()