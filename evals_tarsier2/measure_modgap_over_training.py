import os
import sys
from glob import glob

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shared.utils as su
from utils.video import read_frames_decord


def moggap(df, video_embeds, texts_embeds, text_col):
    X = []
    Y = [] 
    for i in range(len(df)):
        row = df.iloc[i].to_dict()
        zv = video_embeds[row['video_id']]
        zt = texts_embeds[row[text_col]]
        X.append(zv)
        Y.append(zt)
    norm = lambda x: torch.nn.functional.normalize(x, dim=-1)
    X = norm(torch.stack(X))
    Y = norm(torch.stack(Y))
    delta = np.round((X.mean(dim=0) - Y.mean(dim=0)).norm(dim=-1).item(), 2)
    return delta


if __name__ == "__main__":
    data_root = "/scratch/shared/beegfs/piyush/datasets/MSRVTT"
    video_dir = f"{data_root}/videos/all/"

    data = su.io.load_json(f"{data_root}/annotation/msrvtt_test_1k.json")
    df = pd.DataFrame(data)

    df['video_path'] = df.video.apply(lambda x: f"{video_dir}/{x}")
    df['video_path'].apply(os.path.exists).mean()

    text_col = "caption"
    df.shape

    vid2text = {}
    for f in df.video_path.unique():
        vid2text[f] = df[df.video_path == f].iloc[0][text_col]
    len(vid2text)
    
    ckpt_dir = "/work/piyush/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k-stepwise"
    from glob import glob
    from natsort import natsorted
    embed_files = natsorted(
        glob(
            f"{ckpt_dir}/checkpoint-*/merged_checkpoint/embs/"
            "tarsier2+tara_nuanced_retrieval_data-v1_msrvtt_embeddings.pt"
        )
    )
    step_count = [int(e.split("/")[8].split("-")[1]) for e in embed_files]
    
    # For each embedding file, load them, and compute the modality gap
    text_col = "caption"
    steps = [0]
    deltas = [0.49]
    for i in range(len(embed_files)):
        embs = torch.load(embed_files[i])
        video_embeds = {}
        texts_embeds = {}
        
        # Construct dicts for video and text embeddings
        for k in embs:
            if k in set(df[text_col]):
                texts_embeds[k] = embs[k]
            else:
                video_embeds[k] = embs[k]
        delta = moggap(df, video_embeds, texts_embeds, text_col)
        print(f"Step {step_count[i]}: Modality gap = {delta}")
        steps.append(step_count[i])
        deltas.append(delta)

    # Manually annotate losses at these steps
    losses = [
        6.43,
        4.50,
        0.56,
        0.24,
        0.29,
        0.30,
        0.33,
        0.36,
        0.23,
        0.23,
        0.15,
        0.24,
        0.27,
        0.17
    ]
    
    # Get the loss only for the steps that we have deltas
    all_steps = np.arange(0, 140, 10)
    losses = [losses[i] for i in range(len(all_steps)) if all_steps[i] in steps]
    
    loss_color = "royalblue"
    gap_color = "indianred"
    
    # Compute pearson correlation between loss and delta
    import scipy.stats
    correlation, _ = scipy.stats.pearsonr(losses[:len(steps)], deltas[:len(steps)])
    print(f"Pearson correlation between loss and delta: {correlation}")


    # Plot a nice figure
    # Add serif font
    plt.rcParams.update({
        "font.family": "serif",
    })
    
    fig, ax_left = plt.subplots(figsize=(6, 4))
    ax_right = ax_left.twinx()

    line_loss, = ax_left.plot(steps, losses[:len(steps)], "-o", color=loss_color, lw=2, label="Loss", markerfacecolor='none')
    line_gap, = ax_right.plot(steps, deltas, "-s", color=gap_color, lw=2, label="Modality Gap", alpha=0.7, markerfacecolor='none')

    ax_left.set_xlabel("Step")
    ax_left.set_ylabel("Loss", color=loss_color)
    ax_right.set_ylabel("Modality Gap", color=gap_color)
    ax_left.tick_params(axis="y", labelcolor=loss_color)
    ax_right.tick_params(axis="y", labelcolor=gap_color)
    ax_left.set_title("Modality gap and loss over training ($ \\rho = %s $)" % (np.round(correlation, 2)))
    ax_left.grid(alpha=0.25)

    ax_left.legend([line_loss, line_gap], ["Loss", "Modality Gap"], loc="best")
    fig.savefig("modgap_over_training.png", dpi=300, bbox_inches="tight")
    plt.show()