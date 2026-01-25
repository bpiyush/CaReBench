"""
Script to convert all videos in the TEMPO dataset to mp4 format.

This script:
1. Scans the videos directory for all video files
2. Converts non-mp4 files to mp4 format using ffmpeg
3. Optionally removes the original files after successful conversion

Usage:
    python convert_videos_to_mp4.py [--remove-originals]
"""

import os
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


# Video extensions that need conversion (not mp4)
VIDEO_EXTENSIONS = {'.avi', '.mov', '.wmv', '.mpg', '.mpeg', '.3gp', '.m4v', '.mts', '.3g2'}


def get_video_files(video_dir: Path) -> list:
    """Get all video files that need conversion."""
    files_to_convert = []
    
    for file_path in video_dir.iterdir():
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                files_to_convert.append(file_path)
    
    return files_to_convert


def convert_to_mp4(input_path: Path, remove_original: bool = False, timeout: int = 300) -> tuple:
    """
    Convert a video file to mp4 format.
    
    Returns:
        tuple: (success: bool, message: str, input_path: Path)
    """
    output_path = input_path.with_suffix('.mp4')
    
    # Skip if output already exists and has content
    if output_path.exists() and output_path.stat().st_size > 0:
        return True, f"Skipped (mp4 exists): {input_path.name}", input_path
    
    # Clean up any empty/partial output file
    if output_path.exists() and output_path.stat().st_size == 0:
        output_path.unlink()
    
    try:
        # Use ffmpeg to convert with timeout
        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-c:v', 'libx264',      # Video codec
            '-c:a', 'aac',          # Audio codec
            '-preset', 'fast',       # Encoding speed/quality tradeoff
            '-crf', '23',           # Quality (lower = better, 23 is default)
            '-y',                   # Overwrite output
            '-loglevel', 'error',   # Only show errors
            str(output_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        
        if result.returncode != 0:
            # Clean up failed output
            if output_path.exists():
                output_path.unlink()
            return False, f"Failed: {input_path.name} - {result.stderr[:200]}", input_path
        
        # Verify output file exists and has content
        if not output_path.exists() or output_path.stat().st_size == 0:
            return False, f"Failed: {input_path.name} - Output file empty or missing", input_path
        
        # Remove original if requested
        if remove_original:
            input_path.unlink()
            return True, f"Converted and removed: {input_path.name}", input_path
        
        return True, f"Converted: {input_path.name}", input_path
        
    except subprocess.TimeoutExpired:
        # Clean up partial output on timeout
        if output_path.exists():
            try:
                output_path.unlink()
            except:
                pass
        return False, f"Timeout ({timeout}s): {input_path.name}", input_path
    except Exception as e:
        # Clean up on any error
        if output_path.exists():
            try:
                output_path.unlink()
            except:
                pass
        return False, f"Error: {input_path.name} - {str(e)}", input_path


def main():
    parser = argparse.ArgumentParser(description='Convert videos to mp4 format')
    parser.add_argument('--remove-originals', action='store_true',
                        help='Remove original files after successful conversion')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--timeout', type=int, default=300,
                        help='Timeout per file in seconds (default: 300)')
    parser.add_argument('--video-dir', type=str,
                        default='/scratch/shared/beegfs/piyush/datasets/TestOfTime/TEMPO/videos',
                        help='Directory containing videos')
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    
    if not video_dir.exists():
        print(f"Error: Video directory does not exist: {video_dir}")
        return
    
    # Get files to convert
    files_to_convert = get_video_files(video_dir)
    
    if not files_to_convert:
        print("No files need conversion. All videos are already in mp4 format.")
        return
    
    print(f"Found {len(files_to_convert)} files to convert")
    print(f"Timeout per file: {args.timeout}s")
    print(f"Workers: {args.workers}")
    
    # Count by extension
    ext_counts = {}
    for f in files_to_convert:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
    print("Files by extension:")
    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count}")
    print()
    
    # Convert files
    success_count = 0
    skip_count = 0
    fail_count = 0
    failed_files = []
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(convert_to_mp4, f, args.remove_originals, args.timeout): f 
            for f in files_to_convert
        }
        
        with tqdm(total=len(files_to_convert), desc="Converting") as pbar:
            for future in as_completed(futures):
                try:
                    success, message, input_path = future.result(timeout=args.timeout + 60)
                    if success:
                        if "Skipped" in message:
                            skip_count += 1
                        else:
                            success_count += 1
                    else:
                        fail_count += 1
                        failed_files.append((input_path.name, message))
                        tqdm.write(f"FAILED: {input_path.name}")
                except Exception as e:
                    fail_count += 1
                    f = futures[future]
                    failed_files.append((f.name, str(e)))
                    tqdm.write(f"EXCEPTION: {f.name} - {e}")
                pbar.update(1)
    
    # Print report
    print(f"\n{'='*60}")
    print("CONVERSION REPORT")
    print(f"{'='*60}")
    print(f"  Newly converted: {success_count}")
    print(f"  Already existed (skipped): {skip_count}")
    print(f"  Failed: {fail_count}")
    
    # Count total mp4 files
    mp4_count = len(list(video_dir.glob('*.mp4')))
    total_files = len(list(video_dir.iterdir()))
    print(f"\n  Total mp4 files: {mp4_count}")
    print(f"  Total files in directory: {total_files}")
    
    if failed_files:
        print(f"\n{'='*60}")
        print("FAILED FILES:")
        print(f"{'='*60}")
        for fname, reason in failed_files:
            print(f"  {fname}")
            print(f"    Reason: {reason[:100]}...")
        
        # Save failed files list
        failed_list_path = video_dir / "failed_conversions.txt"
        with open(failed_list_path, 'w') as f:
            for fname, reason in failed_files:
                f.write(f"{fname}\t{reason}\n")
        print(f"\nFailed files list saved to: {failed_list_path}")


if __name__ == "__main__":
    main()
