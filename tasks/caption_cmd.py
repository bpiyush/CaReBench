"""Adds captions with TARA to CondensedMovies dataset."""
import os
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="Generate video captions using TARA model")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/datasets/CondensedMovies",
        help="Directory containing video files"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/CondensedMovies/outputs",
        help="Directory to save output captions"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/"
                "nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint",
        help="Path to the captioner model"
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=8,
        help="Number of frames to sample from each video"
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save checkpoint after processing this many videos"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for captions (default: save_dir/tara7b-clip_captions-n=<n_frames>.pt)"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="Start index for video processing (after sorting)"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="End index for video processing (after sorting)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode with visual output"
    )
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load all video files and sort them
    video_files = sorted(glob(f"{args.data_dir}/data_trim/*/*.mkv"))
    print(f"Found {len(video_files)} total video files")
    
    # Apply start and end index filtering
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index if args.start_index is not None else 0
        end = args.end_index if args.end_index is not None else len(video_files)
        video_files = video_files[start:end]
        print(f"Processing videos from index {start} to {end}: {len(video_files)} videos")
    else:
        start = None
        end = None
    
    # Determine output path
    if args.output is None:
        if start is not None or end is not None:
            start_str = start if start is not None else 0
            end_str = end if end is not None else len(video_files)
            captions_path = f"{args.save_dir}/tara7b-clip_captions-n={args.n_frames}-range={start_str}-{end_str}.pt"
        else:
            captions_path = f"{args.save_dir}/tara7b-clip_captions-n={args.n_frames}.pt"
    else:
        captions_path = args.output
    
    # Load existing captions if available
    if os.path.exists(captions_path):
        print("Loading existing captions...")
        captions = torch.load(captions_path)
        print(f"Found {len(captions)} existing captions")
    else:
        captions = {}
    
    # Filter out already processed videos
    videos_to_process = [v for v in video_files if os.path.basename(v) not in captions]
    print(f"Videos to process: {len(videos_to_process)} (out of {len(video_files)} in range)")
    
    if len(videos_to_process) == 0:
        print("All videos in the specified range are already processed!")
    else:
        # Load model
        print(f"Loading model from {args.model_id}...")
        captioner = AutoCaptioner.from_pretrained(args.model_id, device_map='auto')
        su.misc.num_params(captioner.model)
        
        processed_count = 0
        for video_path in su.log.tqdm_iterator(videos_to_process, desc="Generate captions for videos"):
            
            try:
                video_tensor = read_frames_decord(video_path, args.n_frames)
                with torch.no_grad():
                    caption = captioner.describe(video_tensor.unsqueeze(0))[0]
            except:
                print(f"Error reading video {video_path}. Skipping.")
                continue
        
            if args.verbose:
                # display(su.visualize.show_single_video(video_path, label=caption))
                frames = su.video.load_frames_linspace(video_path, n=args.n_frames)
                display(su.visualize.concat_images_with_border(frames))
                display(Markdown(caption))
                break

            video_fname = os.path.basename(video_path)
            captions[video_fname] = caption
            processed_count += 1
            
            # Save checkpoint every save_every videos
            if processed_count % args.save_every == 0:
                torch.save(captions, captions_path)
                print(f"\nCheckpoint saved: {len(captions)} total captions")

        # Final save
        if not args.verbose:
            torch.save(captions, captions_path)
            print(f"Final save complete. Total captions: {len(captions)}")
