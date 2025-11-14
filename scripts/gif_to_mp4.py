#!/usr/bin/env python3
"""
Convert GIF files to MP4 format.

This script takes a folder containing GIF files and converts each one to MP4 format,
saving them to a specified output directory.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def convert_gif_to_mp4(gif_path, output_path):
    """
    Convert a single GIF file to MP4.
    
    Args:
        gif_path: Path to the input GIF file
        output_path: Path where the MP4 file should be saved
        
    Returns:
        True if conversion successful, False otherwise
    """
    try:
        # Use ffmpeg to convert GIF to MP4
        # -y flag overwrites output file if it exists
        # -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" ensures even dimensions (required for some codecs)
        # -pix_fmt yuv420p ensures compatibility with most players
        cmd = [
            'ffmpeg',
            '-i', str(gif_path),
            '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart',
            '-y',  # Overwrite output file if it exists
            str(output_path)
        ]
        
        subprocess.run(cmd, 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting {gif_path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Convert GIF files in a folder to MP4 format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gif_to_mp4.py --input_folder /path/to/gifs --save_dir /path/to/output
  python gif_to_mp4.py -i ./gifs -o ./mp4s
        """
    )
    
    parser.add_argument(
        '--input_folder', '-i',
        type=str,
        required=True,
        help='Path to folder containing GIF files'
    )
    
    parser.add_argument(
        '--save_dir', '-o',
        type=str,
        required=True,
        help='Path to directory where MP4 files will be saved'
    )
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        print("Error: ffmpeg is not installed or not found in PATH.", file=sys.stderr)
        print("Please install ffmpeg: https://ffmpeg.org/download.html", file=sys.stderr)
        sys.exit(1)
    
    # Validate input folder
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder does not exist: {input_folder}", file=sys.stderr)
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: Input path is not a directory: {input_folder}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all GIF files
    gif_files = list(input_folder.glob('*.gif')) + list(input_folder.glob('*.GIF'))
    
    if not gif_files:
        print(f"No GIF files found in {input_folder}")
        sys.exit(0)
    
    print(f"Found {len(gif_files)} GIF file(s) to convert...")
    
    # Convert each GIF to MP4
    successful = 0
    failed = 0
    
    for gif_path in gif_files:
        # Create output filename (same name but with .mp4 extension)
        output_filename = gif_path.stem + '.mp4'
        output_path = save_dir / output_filename
        
        print(f"Converting: {gif_path.name} -> {output_filename}")
        
        if convert_gif_to_mp4(gif_path, output_path):
            successful += 1
            print(f"  ✓ Success: {output_path}")
        else:
            failed += 1
            print(f"  ✗ Failed: {gif_path.name}")
    
    # Summary
    print(f"\nConversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()

