#!/usr/bin/env python3
"""Quick CLIP encode throughput check."""
from __future__ import annotations

import time

from datasets import load_all_videos
from embedders import ClipAvgPoolEmbedder


def main() -> None:
    entries = load_all_videos("cia")[:30]
    print(f"loaded {len(entries)}", flush=True)
    e = ClipAvgPoolEmbedder()
    e.load()
    print("model ready", flush=True)
    t0 = time.time()
    for i, ent in enumerate(entries):
        z = e.encode_video(ent.video_path)
        if i % 5 == 0:
            rate = (i + 1) / max(time.time() - t0, 1e-6)
            print(f"{i} {ent.video_id} {tuple(z.shape)} {rate:.2f}/s", flush=True)
    print(f"done {(time.time() - t0) / len(entries):.3f} s/vid", flush=True)
    e.close()


if __name__ == "__main__":
    main()
