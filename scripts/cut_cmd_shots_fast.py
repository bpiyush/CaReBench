#!/usr/bin/env python3
"""
Fast script to extract and save video shots from CondensedMovies dataset.
Uses direct FFmpeg calls and multiprocessing for maximum speed.
"""

import argparse
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import subprocess
import multiprocessing as mp
from functools import partial
import hashlib


def generate_shot_filename(video_id, start_time, end_time):
    """
    Generate a unique filename for the shot.
    Uses a hash of the video_id to keep filename manageable.
    """
    video_hash = hashlib.md5(video_id.encode()).hexdigest()[:12]
    return f"{video_hash}_{start_time:.2f}_{end_time:.2f}.mp4"


def check_gpu_availability():
    """
    Check if NVIDIA GPU encoding is available.
    """
    try:
        result = subprocess.run(
            ['ffmpeg', '-hide_banner', '-encoders'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return 'h264_nvenc' in result.stdout
    except:
        return False


def process_shot_ffmpeg(video_path, start_time, end_time, output_path, use_gpu=False):
    """
    Extract a shot from video using FFmpeg directly.
    Uses -ss before input for faster seeking.
    """
    try:
        duration = end_time - start_time
        
        # Build FFmpeg command
        # -ss before input is much faster for seeking
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Seek before input (fast)
            '-i', str(video_path),   # Input file
            '-t', str(duration),     # Duration
            '-c:v', 'h264_nvenc' if use_gpu else 'libx264',  # Video codec
            '-preset', 'fast' if use_gpu else 'veryfast',    # Encoding preset
            '-crf', '23',            # Quality (lower = better, 23 is good)
            '-c:a', 'aac',           # Audio codec
            '-b:a', '128k',          # Audio bitrate
            '-movflags', '+faststart',  # Enable streaming
            '-y',                    # Overwrite output
            '-loglevel', 'error',    # Only show errors
            str(output_path)
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout per shot
        )
        
        if result.returncode == 0 and output_path.exists():
            return True, None
        else:
            return False, result.stderr
            
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def process_batch_worker(batch_data, video_dir, save_dir, use_gpu):
    """
    Worker function to process a batch of shots from the same video.
    """
    video_id, shots = batch_data
    video_path = Path(video_dir) / f"{video_id}.mkv"
    
    if not video_path.exists():
        return 0, len(shots), f"Video not found: {video_path}"
    
    successful = 0
    skipped = 0
    errors = []
    
    for shot in shots:
        start_time = shot['st']
        end_time = shot['et']
        
        output_filename = generate_shot_filename(video_id, start_time, end_time)
        output_path = Path(save_dir) / output_filename
        
        # Skip if already exists
        if output_path.exists():
            skipped += 1
            continue
        
        # Process shot
        success, error = process_shot_ffmpeg(video_path, start_time, end_time, output_path, use_gpu)
        
        if success:
            successful += 1
        else:
            skipped += 1
            if error:
                errors.append(f"{video_id} [{start_time}-{end_time}]: {error}")
    
    return successful, skipped, errors


def main():
    parser = argparse.ArgumentParser(description='Fast extraction of video shots from CondensedMovies dataset')
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
                        help='Process only first 100 shots for debugging')
    parser.add_argument('--si', type=int, default=0,
                        help='Start index')
    parser.add_argument('--ei', type=int, default=None,
                        help='End index')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers (default: 8)')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU encoding if available (NVIDIA only)')
    
    args = parser.parse_args()
    
    # Create paths
    video_dir = Path(args.video_dir)
    csv_file = Path(args.csv_file)
    save_dir = Path(args.save_dir)
    
    # Create save directory
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check GPU availability
    use_gpu = False
    if args.gpu:
        use_gpu = check_gpu_availability()
        if use_gpu:
            print("✓ GPU encoding (h264_nvenc) available and enabled")
        else:
            print("✗ GPU encoding not available, falling back to CPU")
    
    # Read CSV file
    print(f"Loading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    print(f"Loaded {len(df)} shots from CSV")
    
    # Limit to 100 shots in debug mode
    if args.debug:
        df = df.head(100)
        print(f"Debug mode: Processing only {len(df)} shots")
    
    # Apply start and end indices
    si = args.si
    ei = args.ei if args.ei is not None else len(df)
    df = df.iloc[si:ei]
    print(f"Processing from index {si} to {ei}: {len(df)} shots")
    
    # Group shots by video_id for batch processing
    print("Grouping shots by video...")
    batches = []
    for video_id, group in df.groupby('video_id'):
        shots = group[['st', 'et']].to_dict('records')
        batches.append((video_id, shots))
    
    print(f"Created {len(batches)} batches from {len(df)} shots")
    print(f"Using {args.workers} parallel workers")
    
    # Process batches in parallel
    successful = 0
    skipped = 0
    all_errors = []
    
    worker_fn = partial(process_batch_worker, 
                       video_dir=str(video_dir), 
                       save_dir=str(save_dir),
                       use_gpu=use_gpu)
    
    with mp.Pool(processes=args.workers) as pool:
        results = list(tqdm(
            pool.imap(worker_fn, batches),
            total=len(batches),
            desc="Processing batches"
        ))
    
    # Aggregate results
    for succ, skip, errors in results:
        successful += succ
        skipped += skip
        if errors:
            all_errors.extend(errors if isinstance(errors, list) else [errors])
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Total processed: {successful + skipped}")
    
    if all_errors:
        print(f"\nErrors encountered: {len(all_errors)}")
        print("First 10 errors:")
        for error in all_errors[:10]:
            print(f"  - {error}")


if __name__ == "__main__":
    main()

