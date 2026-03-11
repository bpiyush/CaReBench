"""Extract the middle frame from each video and save as PNG."""

import os
import glob
from pathlib import Path
from multiprocessing import Pool
from functools import partial

import cv2
from tqdm import tqdm


def extract_mid_frame(video_path: str, output_dir: str) -> str | None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Failed to open: {video_path}"

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return f"No frames in: {video_path}"

    mid = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return f"Failed to read frame {mid} from: {video_path}"

    stem = Path(video_path).stem
    out_path = os.path.join(output_dir, f"{stem}.png")
    cv2.imwrite(out_path, frame)
    return None


def main():
    src = "/scratch/shared/beegfs/piyush/datasets/MSRVTT/videos/all"
    dst = "/scratch/shared/beegfs/piyush/datasets/MSRVTT/videos/mid_frames"
    os.makedirs(dst, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(src, "*.mp4")))
    print(f"Found {len(videos)} videos")

    worker = partial(extract_mid_frame, output_dir=dst)
    with Pool(processes=16) as pool:
        errors = list(tqdm(pool.imap_unordered(worker, videos), total=len(videos)))

    errors = [e for e in errors if e is not None]
    if errors:
        print(f"\n{len(errors)} errors:")
        for e in errors:
            print(f"  {e}")
    else:
        print("All frames extracted successfully.")


if __name__ == "__main__":
    main()
