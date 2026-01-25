"""
Visualize a sample from the TestOfTime (TEMPO) dataset.

Shows frames from both events (before/after) with captions.

Usage:
    python visualize_sample.py [--idx 0] [--n_frames 4] [--save_path output.png]
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import decord


# Default paths
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "tempotl+didemo_test.csv")
DEFAULT_VIDEO_DIR = "/scratch/shared/beegfs/piyush/datasets/TestOfTime/TEMPO/videos"


def load_frames_from_event(video_path: str, start: float, end: float, n_frames: int = 8):
    """
    Load frames from a specific time range of a video.
    
    Args:
        video_path: Path to the video file.
        start: Start time in seconds.
        end: End time in seconds.
        n_frames: Number of frames to extract.
        
    Returns:
        List of PIL Images.
    """
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(video_path, num_threads=1)
    fps = vr.get_avg_fps()
    
    sf = max(int(start * fps), 0)
    ef = min(int(end * fps), len(vr) - 1)
    ef = max(ef, sf)
    
    indices = np.linspace(sf, ef, n_frames, endpoint=True).astype(int)
    frames = [Image.fromarray(vr[i].asnumpy()) for i in indices]
    
    del vr
    return frames


def visualize_sample(
    row: dict,
    video_dir: str,
    n_frames_per_event: int = 8,
    save_path: str = None,
    show_incorrect: bool = False,
):
    """
    Visualize a sample from the TestOfTime dataset.
    
    Args:
        row: A row from the CSV as a dictionary.
        video_dir: Directory containing video files.
        n_frames_per_event: Number of frames to show per event.
        save_path: If provided, save the figure to this path.
        show_incorrect: If True, show the distractor caption instead.
    """
    video_path = os.path.join(video_dir, row['video_id'])
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Load frames from both events
    frames_event1 = load_frames_from_event(
        video_path, 
        row['event1_start'], 
        row['event1_end'], 
        n_frames_per_event
    )
    frames_event2 = load_frames_from_event(
        video_path, 
        row['event2_start'], 
        row['event2_end'], 
        n_frames_per_event
    )
    
    # Create figure
    total_frames = n_frames_per_event * 2
    fig, axes = plt.subplots(2, n_frames_per_event, figsize=(4 * n_frames_per_event, 8))
    
    # Plot event 1 frames (top row)
    for i, frame in enumerate(frames_event1):
        axes[0, i].imshow(frame)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title(f"Event 1: {row['event1_description'][:50]}...", fontsize=10, loc='left')
    
    # Plot event 2 frames (bottom row)
    for i, frame in enumerate(frames_event2):
        axes[1, i].imshow(frame)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(f"Event 2: {row['event2_description'][:50]}...", fontsize=10, loc='left')
    
    # Add caption as figure title
    caption = row['distractor_caption'] if show_incorrect else row['caption']
    caption_type = "INCORRECT (Distractor)" if show_incorrect else "CORRECT"
    
    fig.suptitle(
        f"{caption_type} Caption:\n\"{caption}\"\n\n"
        f"Video: {row['video_id']} | Conjugate: {row['conjugate']}",
        fontsize=12,
        y=1.02,
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
    
    plt.show()
    return fig


def visualize_both_captions(
    row: dict,
    video_dir: str,
    n_frames_per_event: int = 4,
    save_path: str = None,
):
    """
    Visualize a sample showing both correct and incorrect captions side by side.
    
    Args:
        row: A row from the CSV as a dictionary.
        video_dir: Directory containing video files.
        n_frames_per_event: Number of frames to show per event.
        save_path: If provided, save the figure to this path.
    """
    video_path = os.path.join(video_dir, row['video_id'])
    
    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return
    
    # Load frames from both events
    frames_event1 = load_frames_from_event(
        video_path, 
        row['event1_start'], 
        row['event1_end'], 
        n_frames_per_event
    )
    frames_event2 = load_frames_from_event(
        video_path, 
        row['event2_start'], 
        row['event2_end'], 
        n_frames_per_event
    )
    
    # Create figure with 2 rows for events
    fig, axes = plt.subplots(2, n_frames_per_event, figsize=(4 * n_frames_per_event, 9))
    
    # Plot event 1 frames (top row)
    for i, frame in enumerate(frames_event1):
        axes[0, i].imshow(frame)
        axes[0, i].axis('off')
    axes[0, 0].set_ylabel("Event 1", fontsize=12)
    
    # Plot event 2 frames (bottom row)
    for i, frame in enumerate(frames_event2):
        axes[1, i].imshow(frame)
        axes[1, i].axis('off')
    axes[1, 0].set_ylabel("Event 2", fontsize=12)
    
    # Add row labels
    for ax, label in zip(axes[:, 0], [f"Event 1\n({row['event1_start']}-{row['event1_end']}s)", 
                                       f"Event 2\n({row['event2_start']}-{row['event2_end']}s)"]):
        ax.annotate(label, xy=(-0.1, 0.5), xycoords='axes fraction',
                   fontsize=10, ha='right', va='center')
    
    # Create title with both captions
    title = (
        f"Video: {row['video_id']}\n"
        f"Event 1: \"{row['event1_description']}\"\n"
        f"Event 2: \"{row['event2_description']}\"\n\n"
        f"✓ CORRECT: \"{row['caption']}\"\n"
        f"✗ INCORRECT: \"{row['distractor_caption']}\""
    )
    
    fig.suptitle(title, fontsize=11, y=1.08, ha='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved visualization to: {save_path}")
    
    plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize TestOfTime sample')
    parser.add_argument(
        '--csv_path',
        type=str,
        default=DEFAULT_CSV_PATH,
        help='Path to the TEMPO test CSV file',
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help='Directory containing video files',
    )
    parser.add_argument(
        '--idx',
        type=int,
        default=0,
        help='Index of the sample to visualize (default: 0)',
    )
    parser.add_argument(
        '--n_frames',
        type=int,
        default=8,
        help='Number of frames per event to show (default: 4)',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./testoftime-eval/outputs/sample_visualization.png',
        help='Path to save the visualization',
    )
    parser.add_argument(
        '--show_incorrect',
        action='store_true',
        help='Show the incorrect (distractor) caption instead',
    )
    parser.add_argument(
        '--both',
        action='store_true',
        help='Show both correct and incorrect captions',
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Pick a random sample instead of using --idx',
    )
    args = parser.parse_args()
    
    # Load CSV
    print(f"Loading CSV from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Total samples: {len(df)}")
    
    # Select sample
    if args.random:
        idx = np.random.randint(0, len(df))
        print(f"Randomly selected sample index: {idx}")
    else:
        idx = args.idx
    
    row = df.iloc[idx].to_dict()
    
    print(f"\nSample {idx}:")
    print(f"  Video ID: {row['video_id']}")
    print(f"  Conjugate: {row['conjugate']}")
    print(f"  Event 1: {row['event1_description']} ({row['event1_start']}-{row['event1_end']}s)")
    print(f"  Event 2: {row['event2_description']} ({row['event2_start']}-{row['event2_end']}s)")
    print(f"  Correct caption: {row['caption']}")
    print(f"  Distractor caption: {row['distractor_caption']}")
    
    # Create output directory if needed
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Visualize
    if args.both:
        visualize_both_captions(
            row=row,
            video_dir=args.video_dir,
            n_frames_per_event=args.n_frames,
            save_path=args.save_path,
        )
    else:
        visualize_sample(
            row=row,
            video_dir=args.video_dir,
            n_frames_per_event=args.n_frames,
            save_path=args.save_path,
            show_incorrect=args.show_incorrect,
        )


if __name__ == "__main__":
    main()
