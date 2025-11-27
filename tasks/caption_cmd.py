"""Adds captions with TARA to CondensedMovies dataset."""
import os
import sys
from glob import glob
from IPython.display import display, Markdown, Latex

import torch
import pandas as pd
import numpy as np
import decord
import shared.utils as su

from models.modeling_captioners import AutoCaptioner
from utils.video import read_frames_decord


if __name__ == "__main__":
    data_dir = "/datasets/CondensedMovies"

    save_dir = '/scratch/shared/beegfs/piyush/datasets/CondensedMovies/outputs'
    os.makedirs(save_dir, exist_ok=True)
    
    # Load all video files
    video_files = glob(f"{data_dir}/data_trim/*/*.mkv")
    print(f"Found {len(video_files)} video files")

    verbose = False
    n_frames = 8
    captions_path = f"{save_dir}/tara7b-clip_captions-n={n_frames}.pt"
    save_every = 1000

    # Load model
    model_id = "/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/"\
        "nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint"
    captioner = AutoCaptioner.from_pretrained(model_id, device_map='auto')
    su.misc.num_params(captioner.model)


    # Load existing captions if available
    if os.path.exists(captions_path):
        print("Loading existing captions...")
        captions = torch.load(captions_path)
        print(f"Found {len(captions)} existing captions")
    else:
        captions = {}
    
    # Filter out already processed videos
    videos_to_process = [v for v in video_files if os.path.basename(v) not in captions]
    print(f"Videos to process: {len(videos_to_process)} (out of {len(video_files)} total)")
    
    if len(videos_to_process) == 0:
        print("All videos already processed!")
    else:
        processed_count = 0
        for video_path in su.log.tqdm_iterator(videos_to_process, desc="Generate captions for videos"):
        
            video_tensor = read_frames_decord(video_path, n_frames)
            with torch.no_grad():
                caption = captioner.describe(video_tensor.unsqueeze(0))[0]
        
            if verbose:
                # display(su.visualize.show_single_video(video_path, label=caption))
                frames = su.video.load_frames_linspace(video_path, n=n_frames)
                display(su.visualize.concat_images_with_border(frames))
                display(Markdown(caption))
                break

            video_fname = os.path.basename(video_path)
            captions[video_fname] = caption
            processed_count += 1
            
            # Save checkpoint every save_every videos
            if processed_count % save_every == 0:
                torch.save(captions, captions_path)
                print(f"\nCheckpoint saved: {len(captions)} total captions")

        # Final save
        if not verbose:
            torch.save(captions, captions_path)
            print(f"Final save complete. Total captions: {len(captions)}")
