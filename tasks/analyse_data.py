import os
import sys

import shared.utils as su

import pandas as pd
import numpy as np
import clip
import torch


def encode_sentences(sentences):
    """
    Encode a list of sentences using CLIP.
    
    Args:
        sentences: List of strings
        
    Returns:
        torch.Tensor of shape [B, D] where B is batch size and D is embedding dimension (512 for ViT-B/16)
    """
    # Tokenize sentences
    text_tokens = clip.tokenize(sentences, truncate=True, context_length=77).to(device)
    
    # Encode text
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        # Normalize embeddings (CLIP uses normalized features)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        text_embeddings = text_embeddings.cpu().float()
    
    return text_embeddings  # Shape: [B, 512]


def encode_sentences_batchwise(sentences, model, device, batch_size=16):
    """
    Encode a list of sentences using CLIP in a batchwise manner.
    
    Args:
        sentences: List of strings
        model: CLIP model
        device: Device to use
        batch_size: Batch size
        
    Returns:
        torch.Tensor of shape [B, D] where B is batch size and D is embedding dimension (512 for ViT-B/16)
    """
    text_embeddings = []
    for i in su.log.tqdm_iterator(range(0, len(sentences), batch_size), desc="Encoding sentences"):
        batch = sentences[i:i+batch_size]
        text_embeddings.append(encode_sentences(batch, model, device))
    text_embeddings = torch.cat(text_embeddings, dim=0)
    return text_embeddings


if __name__ == "__main__":
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model = model.to(device)
    su.misc.num_params(model)

    # Example usage
    sentences = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
    embeddings = encode_sentences(sentences)
    print(f"Embeddings shape: {embeddings.shape}")  # [3, 512]

    # Load CSV
    csv_path = "/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/final-10112025/nli_9000+ego_1000+subj_replaced-seed_42.csv"
    df = pd.read_csv(csv_path)
    
    z_sent0 = encode_sentences(df.sent0.tolist())
    z_sent1 = encode_sentences(df.sent1.tolist())
    z_hard_neg = encode_sentences(df.hard_neg.tolist())
    import ipdb; ipdb.set_trace()
    print(f"Sent0 embeddings shape: {z_sent0.shape}")
    print(f"Sent1 embeddings shape: {z_sent1.shape}")
    
    indices_nli = np.where(df['source'] == 'nli')[0]
    indices_ego = np.where(df['source'] == 'ego4d')[0]
    
    z_nli = torch.cat([z_sent0[indices_nli], z_sent1[indices_nli], z_hard_neg[indices_nli]], dim=0)
    z_ego = torch.cat([z_sent0[indices_ego], z_sent1[indices_ego], z_hard_neg[indices_ego]], dim=0)
    labels = ['NLI'] * len(indices_nli) + ['Ego4D'] * len(indices_ego)
    
    z = torch.cat([z_nli, z_ego], dim=0)
    tsne = su.visualize.reduce_dim(z.numpy(), method="tsne")
    
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    su.visualize.show_projections_with_labels(
        tsne,
        labels,
        title="NLI vs Ego4D",
        cmap='coolwarm',
        legend=True,
        legend_ncol=1,
    )
    plt.savefig("nli_vs_ego4d.png")
    
    indices_subj_replaced = np.where(df['source'] == 'subj_replaced')
    
    # Compute cosine similarity
    similarity = torch.nn.functional.cosine_similarity(z_sent0, z_sent1, dim=-1)
    print(f"Similarity shape: {similarity.shape}")
    print(f"Similarity: {similarity}")
    
    # Save similarity