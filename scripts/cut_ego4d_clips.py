#!/usr/bin/env python3
"""
Script to extract, downscale and save video clips from full videos.
"""

import argparse
import os
from pathlib import Path
from moviepy.editor import VideoFileClip
from tqdm import tqdm


def parse_clip_id(clip_id):
    """
    Parse clip ID into video_id, start_time, and end_time.
    Format: {video_id}_{start}_{end}
    """
    parts = clip_id.strip().split('_')
    if len(parts) < 3:
        raise ValueError(f"Invalid clip ID format: {clip_id}")
    
    # Video ID is all parts except the last two (which are start and end times)
    video_id = '_'.join(parts[:-2])
    start_time = int(parts[-2])
    end_time = int(parts[-1])
    
    return video_id, start_time, end_time


def process_clip(video_path, start_time, end_time, output_path, target_width):
    """
    Extract a clip from video, downscale it, and save.
    """
    try:
        # Load video
        video = VideoFileClip(str(video_path))
        
        # Check if time range is valid
        if end_time > video.duration:
            print(f"Warning: End time {end_time}s exceeds video duration {video.duration}s for {video_path.name}")
            video.close()
            return False
        
        # Extract clip
        clip = video.subclip(start_time, end_time)
        
        # Calculate new height to maintain aspect ratio
        original_width, original_height = clip.size
        target_height = int((target_width / original_width) * original_height)
        
        # Resize clip
        resized_clip = clip.resize(width=target_width, height=target_height)
        
        # Save clip
        resized_clip.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            verbose=False,
            logger=None
        )
        
        # Clean up
        resized_clip.close()
        clip.close()
        video.close()
        
        return True
        
    except Exception as e:
        print(f"Error processing clip: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Extract and downscale video clips')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing source videos')
    parser.add_argument('--anno_file', type=str, required=True,
                        help='Text file with clip IDs')
    parser.add_argument('--width', type=int, default=360,
                        help='Target width for downscaling (default: 360)')
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory to save processed clips')
    parser.add_argument('--debug', action='store_true',
                        help='Process only first 10 clips for debugging')
    
    args = parser.parse_args()
    
    # Create paths
    video_dir = Path(args.video_dir)
    anno_file = Path(args.anno_file)
    save_dir = Path(args.save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Read annotation file
    with open(anno_file, 'r') as f:
        clip_ids = [line.strip() for line in f if line.strip()]
    
    # Limit to 10 clips in debug mode
    if args.debug:
        clip_ids = clip_ids[:10]
        print(f"Debug mode: Processing only {len(clip_ids)} clips")
    
    print(f"Processing {len(clip_ids)} clips...")
    
    # Process each clip
    successful = 0
    skipped = 0
    
    for clip_id in tqdm(clip_ids, desc="Processing clips"):
        try:
            # Parse clip ID
            video_id, start_time, end_time = parse_clip_id(clip_id)
            
            # Find video file
            video_path = video_dir / f"{video_id}.mp4"
            
            if not video_path.exists():
                print(f"\nVideo does not exist: {video_path}")
                skipped += 1
                continue
            
            # Output path
            output_path = save_dir / f"{clip_id}.mp4"
            
            # Process clip
            if process_clip(video_path, start_time, end_time, output_path, args.width):
                successful += 1
            else:
                skipped += 1
                
        except Exception as e:
            print(f"\nError processing {clip_id}: {e}")
            skipped += 1
    
    print(f"\nProcessing complete!")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()