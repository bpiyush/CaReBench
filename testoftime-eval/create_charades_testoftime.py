"""
Create a TestOfTime-style evaluation set from Charades dataset.

For each video with multiple clips:
- Pick two clips A and B (where A starts before B)
- Create "A before B" sample with correct and distractor captions
- Create "B after A" sample with correct and distractor captions

Usage:
    python create_charades_testoftime.py [--n_videos 500]
"""
import os
import pandas as pd
import numpy as np
from collections import defaultdict
import argparse


# Paths
CHARADES_CSV = "/scratch/shared/beegfs/piyush/datasets/Charades/metadata_annotations/Charades_v1_clips_all.csv"
CHARADES_CLIP_DIR = "/scratch/shared/beegfs/piyush/datasets/Charades/Charades_v1_480_cut_clips"
OUTPUT_DIR = os.path.dirname(__file__)


def get_clip_filename(video_id: str, start_time: float, end_time: float) -> str:
    """Get the clip filename from video ID and timestamps."""
    return f"{video_id}_{start_time}_{end_time}.mp4"


def clips_overlap(clip_a: dict, clip_b: dict) -> bool:
    """Check if two clips have overlapping time ranges."""
    # Overlap if: start_a < end_b AND start_b < end_a
    return clip_a['start_time'] < clip_b['end_time'] and clip_b['start_time'] < clip_a['end_time']


def find_non_overlapping_pair(clips: list) -> tuple:
    """
    Find a pair of non-overlapping clips from a list.
    
    Args:
        clips: List of clip dictionaries sorted by start_time.
        
    Returns:
        Tuple of (clip_a, clip_b) if found, else (None, None).
    """
    # Try all pairs to find non-overlapping ones
    for i in range(len(clips)):
        for j in range(i + 1, len(clips)):
            clip_a = clips[i]
            clip_b = clips[j]
            if not clips_overlap(clip_a, clip_b):
                return clip_a, clip_b
    return None, None


def create_testoftime_samples(df: pd.DataFrame, n_videos: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Create TestOfTime-style samples from Charades clips.
    
    Args:
        df: DataFrame with Charades clip annotations.
        n_videos: Number of videos to sample (will create 2 samples per video pair).
        seed: Random seed for reproducibility.
        
    Returns:
        DataFrame with TestOfTime-style samples.
    """
    np.random.seed(seed)
    
    # Group clips by video ID
    video_clips = defaultdict(list)
    for _, row in df.iterrows():
        video_clips[row['id']].append(row.to_dict())
    
    # Filter videos with at least 2 clips
    valid_videos = {vid: clips for vid, clips in video_clips.items() if len(clips) >= 2}
    print(f"Videos with at least 2 clips: {len(valid_videos)}")
    
    # Randomly shuffle video IDs
    all_video_ids = list(valid_videos.keys())
    np.random.shuffle(all_video_ids)
    
    samples = []
    videos_used = 0
    
    for video_id in all_video_ids:
        if videos_used >= n_videos:
            break
            
        clips = valid_videos[video_id]
        
        # Sort clips by start time to ensure temporal order
        clips_sorted = sorted(clips, key=lambda x: x['start_time'])
        
        # Find a non-overlapping pair
        clip_a, clip_b = find_non_overlapping_pair(clips_sorted)
        
        if clip_a is None or clip_b is None:
            # No non-overlapping pair found for this video
            continue
        
        # Get clip filenames
        clip_a_file = get_clip_filename(video_id, clip_a['start_time'], clip_a['end_time'])
        clip_b_file = get_clip_filename(video_id, clip_b['start_time'], clip_b['end_time'])
        
        # Check if clip files exist
        clip_a_path = os.path.join(CHARADES_CLIP_DIR, clip_a_file)
        clip_b_path = os.path.join(CHARADES_CLIP_DIR, clip_b_file)
        
        if not os.path.exists(clip_a_path) or not os.path.exists(clip_b_path):
            continue
        
        # Get action descriptions
        action_a = clip_a['cls_name']
        action_b = clip_b['cls_name']
        
        # Skip if same action (not interesting for temporal understanding)
        if action_a == action_b:
            continue
        
        # Create "A before B" sample
        sample_before = {
            'video_id': video_id,
            'conjugate': 'before',
            'event1_description': action_a,
            'event2_description': action_b,
            'caption': f"{action_a} before {action_b}",
            'distractor_caption': f"{action_b} before {action_a}",
            'event1_clip': clip_a_file,
            'event2_clip': clip_b_file,
            'event1_start': clip_a['start_time'],
            'event1_end': clip_a['end_time'],
            'event2_start': clip_b['start_time'],
            'event2_end': clip_b['end_time'],
        }
        samples.append(sample_before)
        
        # Create "B after A" sample
        sample_after = {
            'video_id': video_id,
            'conjugate': 'after',
            'event1_description': action_a,
            'event2_description': action_b,
            'caption': f"{action_b} after {action_a}",
            'distractor_caption': f"{action_a} after {action_b}",
            'event1_clip': clip_a_file,
            'event2_clip': clip_b_file,
            'event1_start': clip_a['start_time'],
            'event1_end': clip_a['end_time'],
            'event2_start': clip_b['start_time'],
            'event2_end': clip_b['end_time'],
        }
        samples.append(sample_after)
        
        videos_used += 1
    
    print(f"Successfully created samples from {videos_used} videos")
    return pd.DataFrame(samples)


def main():
    parser = argparse.ArgumentParser(description='Create Charades TestOfTime evaluation set')
    parser.add_argument(
        '--csv_path',
        type=str,
        default=CHARADES_CSV,
        help='Path to Charades clips CSV',
    )
    parser.add_argument(
        '--n_videos',
        type=int,
        default=500,
        help='Number of videos to sample (default: 500)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)',
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=os.path.join(OUTPUT_DIR, 'charades_testoftime.csv'),
        help='Output CSV path',
    )
    args = parser.parse_args()
    
    # Load Charades CSV
    print(f"Loading Charades CSV from: {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    print(f"Total clips: {len(df)}")
    print(f"Unique videos: {df['id'].nunique()}")
    
    # Create TestOfTime samples
    samples_df = create_testoftime_samples(df, n_videos=args.n_videos, seed=args.seed)
    
    print(f"\nCreated {len(samples_df)} samples")
    print(f"  - 'before' samples: {len(samples_df[samples_df['conjugate'] == 'before'])}")
    print(f"  - 'after' samples: {len(samples_df[samples_df['conjugate'] == 'after'])}")
    print(f"  - Unique videos: {samples_df['video_id'].nunique()}")
    
    # Save to CSV
    samples_df.to_csv(args.output_path, index=False)
    print(f"\nSaved to: {args.output_path}")
    
    # Show sample rows
    print("\nSample rows:")
    print(samples_df.head(4).to_string())


if __name__ == "__main__":
    main()
