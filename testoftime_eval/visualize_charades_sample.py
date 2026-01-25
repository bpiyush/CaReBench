"""
Visualize a sample from the Charades TestOfTime dataset.

Shows frames from both clips (event1 and event2) with captions.

Usage:
    python visualize_charades_sample.py [--idx 0] [--n_frames 8] [--save_path output.png]
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
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "charades_testoftime.csv")
DEFAULT_CLIP_DIR = "/scratch/shared/beegfs/piyush/datasets/Charades/Charades_v1_480_cut_clips"


def load_frames_from_clip(clip_path: str, n_frames: int = 8):
    """
    Load frames from a clip file.
    
    Args:
        clip_path: Path to the clip file.
        n_frames: Number of frames to extract.
        
    Returns:
        List of PIL Images.
    """
    decord.bridge.set_bridge('native')
    vr = decord.VideoReader(clip_path, num_threads=1)
    
    indices = np.linspace(0, len(vr) - 1, n_frames, endpoint=True).astype(int)
    frames = [Image.fromarray(vr[i].asnumpy()) for i in indices]
    
    del vr
    return frames


def visualize_sample(
    row: dict,
    clip_dir: str,
    n_frames_per_clip: int = 8,
    save_path: str = None,
    show_incorrect: bool = False,
):
    """
    Visualize a sample from the Charades TestOfTime dataset.
    
    Args:
        row: A row from the CSV as a dictionary.
        clip_dir: Directory containing clip files.
        n_frames_per_clip: Number of frames to show per clip.
        save_path: If provided, save the figure to this path.
        show_incorrect: If True, show the distractor caption instead.
    """
    clip1_path = os.path.join(clip_dir, row['event1_clip'])
    clip2_path = os.path.join(clip_dir, row['event2_clip'])
    
    if not os.path.exists(clip1_path):
        print(f"Clip not found: {clip1_path}")
        return
    if not os.path.exists(clip2_path):
        print(f"Clip not found: {clip2_path}")
        return
    
    # Load frames from both clips
    frames_clip1 = load_frames_from_clip(clip1_path, n_frames_per_clip)
    frames_clip2 = load_frames_from_clip(clip2_path, n_frames_per_clip)
    
    # Create figure
    fig, axes = plt.subplots(2, n_frames_per_clip, figsize=(4 * n_frames_per_clip, 8))
    
    # Plot clip 1 frames (top row)
    for i, frame in enumerate(frames_clip1):
        axes[0, i].imshow(frame)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title(f"Event 1: {row['event1_description'][:50]}...", fontsize=10, loc='left')
    
    # Plot clip 2 frames (bottom row)
    for i, frame in enumerate(frames_clip2):
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
    clip_dir: str,
    n_frames_per_clip: int = 8,
    save_path: str = None,
):
    """
    Visualize a sample showing both correct and incorrect captions.
    
    Args:
        row: A row from the CSV as a dictionary.
        clip_dir: Directory containing clip files.
        n_frames_per_clip: Number of frames to show per clip.
        save_path: If provided, save the figure to this path.
    """
    clip1_path = os.path.join(clip_dir, row['event1_clip'])
    clip2_path = os.path.join(clip_dir, row['event2_clip'])
    
    if not os.path.exists(clip1_path):
        print(f"Clip not found: {clip1_path}")
        return
    if not os.path.exists(clip2_path):
        print(f"Clip not found: {clip2_path}")
        return
    
    # Load frames from both clips
    frames_clip1 = load_frames_from_clip(clip1_path, n_frames_per_clip)
    frames_clip2 = load_frames_from_clip(clip2_path, n_frames_per_clip)
    
    # Create figure with 2 rows for clips
    fig, axes = plt.subplots(2, n_frames_per_clip, figsize=(4 * n_frames_per_clip, 9))
    
    # Plot clip 1 frames (top row)
    for i, frame in enumerate(frames_clip1):
        axes[0, i].imshow(frame)
        axes[0, i].axis('off')
    
    # Plot clip 2 frames (bottom row)
    for i, frame in enumerate(frames_clip2):
        axes[1, i].imshow(frame)
        axes[1, i].axis('off')
    
    # Add row labels
    for ax, label in zip(axes[:, 0], [f"Event 1\n({row['event1_clip']})", 
                                       f"Event 2\n({row['event2_clip']})"]):
        ax.annotate(label, xy=(-0.1, 0.5), xycoords='axes fraction',
                   fontsize=9, ha='right', va='center')
    
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
    parser = argparse.ArgumentParser(description='Visualize Charades TestOfTime sample')
    parser.add_argument(
        '--csv_path',
        type=str,
        default=DEFAULT_CSV_PATH,
        help='Path to the Charades TestOfTime CSV file',
    )
    parser.add_argument(
        '--clip_dir',
        type=str,
        default=DEFAULT_CLIP_DIR,
        help='Directory containing clip files',
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
        help='Number of frames per clip to show (default: 8)',
    )
    parser.add_argument(
        '--save_path',
        type=str,
        default='./testoftime-eval/outputs/charades_sample_visualization.png',
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
    print(f"  Event 1: {row['event1_description']} (clip: {row['event1_clip']})")
    print(f"  Event 2: {row['event2_description']} (clip: {row['event2_clip']})")
    print(f"  Correct caption: {row['caption']}")
    print(f"  Distractor caption: {row['distractor_caption']}")
    
    # Create output directory if needed
    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # Visualize
    if args.both:
        visualize_both_captions(
            row=row,
            clip_dir=args.clip_dir,
            n_frames_per_clip=args.n_frames,
            save_path=args.save_path,
        )
    else:
        visualize_sample(
            row=row,
            clip_dir=args.clip_dir,
            n_frames_per_clip=args.n_frames,
            save_path=args.save_path,
            show_incorrect=args.show_incorrect,
        )


if __name__ == "__main__":
    main()
