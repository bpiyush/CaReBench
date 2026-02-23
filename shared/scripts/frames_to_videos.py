#!/usr/bin/env python3

import argparse
import os
import re
from pathlib import Path
from typing import Iterable

import cv2
from tqdm import tqdm
from joblib import Parallel, delayed
from joblib import parallel as joblib_parallel
from contextlib import contextmanager


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def natural_key(path: Path) -> list:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(p) if p.isdigit() else p for p in parts]


def _immediate_subdirs(parent: Path) -> list[Path]:
    """List immediate child directories (one level only, no recursion)."""
    try:
        return sorted(
            Path(e.path) for e in os.scandir(parent) if e.is_dir(follow_symlinks=False)
        )
    except PermissionError:
        return []


def _leaf_dirs(root: Path) -> list[Path]:
    """Walk *root* and return directories that contain no sub-directories (leaf dirs)."""
    leaves: list[Path] = []
    for dirpath, dirnames, _filenames in os.walk(root):
        if not dirnames:
            leaves.append(Path(dirpath))
    leaves.sort()
    return leaves


def list_frame_dirs(frames_root: Path) -> list[Path]:
    """
    Use the known tree layout instead of rglob:
      frames_root/
        Breakfast/<video_id>/   <- one level deep
        HMDB51/<video_id>/
        K700/<video_id>/
        SSv2/<video_id>/
        UCF101/<video_id>/
        data/.../<video_id>/    <- nested, find leaf dirs
    """
    frame_dirs: list[Path] = []
    for entry in _immediate_subdirs(frames_root):
        if entry.name == "data":
            frame_dirs.extend(_leaf_dirs(entry))
        else:
            frame_dirs.extend(_immediate_subdirs(entry))
    return frame_dirs


def list_images(frame_dir: Path) -> list[Path]:
    images = [p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort(key=natural_key)
    return images


def ensure_even(x: int) -> int:
    return x if x % 2 == 0 else x - 1


def output_path_for(frame_dir: Path, frames_root: Path, videos_root: Path, ext: str) -> Path:
    rel_dir = frame_dir.relative_to(frames_root)
    return (videos_root / rel_dir).with_suffix(f".{ext}")


def write_video_from_frames(
    frame_paths: Iterable[Path],
    output_path: Path,
    fps: float,
    overwrite: bool = False,
) -> bool:
    if output_path.exists() and output_path.stat().st_size > 0 and not overwrite:
        return True

    frame_paths = list(frame_paths)
    if not frame_paths:
        return False

    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        return False

    h, w = first.shape[:2]
    w, h = ensure_even(w), ensure_even(h)
    if w <= 0 or h <= 0:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        return False

    ok = True
    try:
        for p in frame_paths:
            frame = cv2.imread(str(p))
            if frame is None:
                ok = False
                break
            if frame.shape[1] != w or frame.shape[0] != h:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    except Exception:
        ok = False
    finally:
        writer.release()

    if ok and output_path.exists() and output_path.stat().st_size > 0:
        return True

    try:
        if output_path.exists():
            output_path.unlink()
    except Exception:
        pass
    return False


@contextmanager
def tqdm_joblib(tqdm_object):
    """Patch joblib to update a tqdm progress bar on batch completion."""
    class TqdmBatchCompletionCallback(joblib_parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_cb = joblib_parallel.BatchCompletionCallBack
    joblib_parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib_parallel.BatchCompletionCallBack = old_cb
        tqdm_object.close()


def _process_one(frame_dir: Path, frames_root: Path, videos_root: Path, ext: str, fps: float, overwrite: bool) -> int:
    """Worker: returns 1=converted, 0=skipped, -1=failed."""
    images = list_images(frame_dir)
    if not images:
        return 0
    out_path = output_path_for(frame_dir, frames_root, videos_root, ext)
    already_exists = out_path.exists() and out_path.stat().st_size > 0
    ok = write_video_from_frames(images, out_path, fps=fps, overwrite=overwrite)
    if ok:
        return 0 if (already_exists and not overwrite) else 1
    return -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively convert frame folders to MP4 videos with mirrored directory structure."
    )
    parser.add_argument(
        "--frames_root",
        type=Path,
        default=Path("/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames"),
        help="Root directory containing frame folders.",
    )
    parser.add_argument(
        "--videos_root",
        type=Path,
        default=Path("/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/videos"),
        help="Root directory where MP4s will be saved.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=1.0,
        help="Output video FPS.",
    )
    parser.add_argument(
        "--ext",
        type=str,
        default="mp4",
        help="Output video extension (without leading dot).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (-1 = all cores).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Process only first 10 frame folders.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frames_root: Path = args.frames_root
    videos_root: Path = args.videos_root
    fps: float = float(args.fps)
    ext: str = args.ext.lstrip(".")

    if not frames_root.is_dir():
        print(f"ERROR: frames_root not found or not a directory: {frames_root}")
        return 1
    if fps <= 0:
        print("ERROR: fps must be > 0.")
        return 1

    print("Discovering frame folders ...")
    frame_dirs = list_frame_dirs(frames_root)
    if args.debug:
        frame_dirs = frame_dirs[:10]

    if not frame_dirs:
        print("No frame folders found.")
        return 0

    print(f"Found {len(frame_dirs)} frame folders. Converting with n_jobs={args.n_jobs} ...")

    with tqdm_joblib(tqdm(total=len(frame_dirs), desc="Converting", unit="video")):
        results = Parallel(n_jobs=args.n_jobs, backend="loky")(
            delayed(_process_one)(fd, frames_root, videos_root, ext, fps, args.overwrite)
            for fd in frame_dirs
        )

    converted = sum(1 for r in results if r == 1)
    skipped = sum(1 for r in results if r == 0)
    failed = sum(1 for r in results if r == -1)

    print(f"Done. converted={converted}, skipped={skipped}, failed={failed}")
    return 2 if failed > 0 else 0


if __name__ == "__main__":
    raise SystemExit(main())
