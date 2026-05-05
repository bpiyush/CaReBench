#!/usr/bin/env python3
"""
Reverse all videos in a folder and save them alongside originals.

Output naming:
    input.ext -> input-reversed.ext
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Common container extensions. Extend this set if needed.
VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".flv",
}


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available in PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def reverse_video(input_path: Path, output_path: Path, overwrite: bool = False) -> None:
    """
    Reverse video and audio streams with ffmpeg.

    Falls back to video-only reverse when an audio stream is absent.
    """
    overwrite_flag = "-y" if overwrite else "-n"

    # First attempt: reverse both video and audio.
    cmd_full = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        overwrite_flag,
        "-i",
        str(input_path),
        "-vf",
        "reverse",
        "-af",
        "areverse",
        str(output_path),
    ]

    try:
        subprocess.run(cmd_full, check=True)
        return
    except subprocess.CalledProcessError:
        # Fallback for videos that do not have an audio stream.
        cmd_video_only = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            overwrite_flag,
            "-i",
            str(input_path),
            "-vf",
            "reverse",
            str(output_path),
        ]
        subprocess.run(cmd_video_only, check=True)


def list_video_files(folder: Path) -> list[Path]:
    """List files in folder that look like videos by extension."""
    return sorted(
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in VIDEO_EXTENSIONS
        and not p.stem.endswith("-reversed")
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Time-reverse videos in a folder and save as *-reversed.<ext>"
    )
    parser.add_argument(
        "folder",
        type=str,
        help="Folder containing input videos",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing reversed outputs if they already exist",
    )
    args = parser.parse_args()

    if not check_ffmpeg():
        print("Error: ffmpeg is not installed or not found in PATH.", file=sys.stderr)
        sys.exit(1)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        print(f"Error: not a valid folder: {folder}", file=sys.stderr)
        sys.exit(1)

    videos = list_video_files(folder)
    if not videos:
        print(f"No video files found in {folder}")
        return

    print(f"Found {len(videos)} video(s) in {folder}")
    success = 0
    skipped = 0
    failed = 0

    for video_path in videos:
        output_path = video_path.with_name(f"{video_path.stem}-reversed{video_path.suffix}")

        if output_path.exists() and not args.overwrite:
            print(f"Skipping existing file: {output_path.name}")
            skipped += 1
            continue

        print(f"Reversing: {video_path.name} -> {output_path.name}")
        try:
            reverse_video(video_path, output_path, overwrite=args.overwrite)
            success += 1
        except subprocess.CalledProcessError as exc:
            print(f"Failed: {video_path.name} ({exc})", file=sys.stderr)
            failed += 1

    print("\nDone.")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
