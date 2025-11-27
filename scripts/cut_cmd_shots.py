#!/usr/bin/env python3
"""
Script to extract and save video shots from CondensedMovies dataset.
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import hashlib


def generate_shot_filename(video_id, start_time, end_time):
    """
    Generate a unique filename for the shot.
    Uses a hash of the video_id to keep filename manageable.
    """
    # Create a hash of the video_id to shorten it
    video_hash = hashlib.md5(video_id.encode()).hexdigest()[:12]
    # Format: hash_start_end.mp4
    return f"{video_hash}_{start_time:.2f}_{end_time:.2f}.mp4"


def process_shot(video_path, start_time, end_time, output_path):
    """
    Extract a shot from video and save.
    """
    try:
        # Load video
        video = VideoFileClip(str(video_path))
        
        # Check if time range is valid
        if end_time > video.duration:
            print(f"\nWarning: End time {end_time}s exceeds video duration {video.duration}s for {video_path.name}")
            end_time = video.duration
        
        if start_time >= video.duration:
            print(f"\nWarning: Start time {start_time}s exceeds video duration {video.duration}s for {video_path.name}")
            video.close()
            return False
        
        # Extract clip
        clip = video.subclip(start_time, end_time)
        
        # Save clip
        clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None,
            threads=4,
            preset='ultrafast'
        )
        
        # Clean up
        clip.close()
        video.close()
        
        return True
        
    except Exception as e:
        print(f"\nError processing shot: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract video shots from CondensedMovies dataset')
    parser.add_argument('--csv_file', type=str, 
                        default='/scratch/shared/beegfs/piyush/datasets/CondensedMovies/metadata/shots.csv',
                        help='CSV file with shot annotations')
    parser.add_argument('--video_dir', type=str, 
                        default='/datasets/CondensedMovies/data_trim/',
                        help='Base directory containing source videos')
    parser.add_argument('--save_dir', type=str, 
                        default='/scratch/shared/beegfs/piyush/datasets/CondensedMovies/shots/',
                        help='Directory to save processed shots')
    parser.add_argument('--debug', action='store_true',
                        help='Process only first 10 shots for debugging')
    parser.add_argument('--si', type=int, default=0,
                        help='Start index')
    parser.add_argument('--ei', type=int, default=None,
                        help='End index')
    
    args = parser.parse_args()
    
    # Create paths
    video_dir = Path(args.video_dir)
    csv_file = Path(args.csv_file)
    save_dir = Path(args.save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Read CSV file
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} shots from CSV")
    print(f"Columns: {list(df.columns)}")
    
    # Limit to 10 shots in debug mode
    if args.debug:
        df = df.head(100)
        print(f"Debug mode: Processing only {len(df)} shots")
    
    # Apply start and end indices
    si = args.si
    ei = args.ei if args.ei is not None else len(df)
    df = df.iloc[si:ei]
    print(f"Processing from index {si} to {ei}: {len(df)} shots")
    
    # Process each shot
    successful = 0
    skipped = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing shots"):
        try:
            video_id = row['video_id']
            start_time = float(row['st'])
            end_time = float(row['et'])
            
            # Construct full video path
            video_path = video_dir / f"{video_id}.mkv"
            
            if not video_path.exists():
                print(f"\nVideo does not exist: {video_path}")
                skipped += 1
                continue
            
            # Generate output filename
            output_filename = generate_shot_filename(video_id, start_time, end_time)
            output_path = save_dir / output_filename
            
            if output_path.exists():
                # print(f"\nOutput already exists: {output_path}")
                skipped += 1
                continue
            
            # Process shot
            if process_shot(video_path, start_time, end_time, output_path):
                successful += 1
            else:
                skipped += 1
                
        except Exception as e:
            print(f"\nError processing row {idx}: {e}")
            skipped += 1
    
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()

