#!/usr/bin/env python3
"""Offline test: 3×3 video grid → GIF with transparent gaps (no labels/cards).

Usage:
  python make_transparent_gif_grid.py -o /path/out.gif video1.mp4 ... video9.mp4
  # or auto-pick 9 MSRVTT previews:
  python make_transparent_gif_grid.py --demo -o /path/out.gif
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Allow importing gif_grid from the package root when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from gif_grid import build_gif  # noqa: E402


def demo_videos() -> list[Path]:
    root = Path("/work/piyush/experiments/TARA-demo/cache/previews/tara__msrvtt")
    vids = sorted(root.glob("*.mp4"))[:9]
    if len(vids) < 9:
        raise FileNotFoundError(f"Need 9 preview mp4s under {root}")
    return vids


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("videos", nargs="*", type=Path, help="Up to 9 video paths")
    p.add_argument("--demo", action="store_true", help="Use 9 cached MSRVTT previews")
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("/work/piyush/experiments/TARA-demo/cache/clips/test_grid_transparent.gif"),
    )
    p.add_argument("--duration", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--cell-w", type=int, default=320)
    p.add_argument("--cell-h", type=int, default=200)
    p.add_argument("--gap", type=int, default=8)
    p.add_argument(
        "--still",
        type=Path,
        default=None,
        help="Also write a transparent PNG still for easy review",
    )
    args = p.parse_args()

    vids = demo_videos() if args.demo else list(args.videos)
    if not vids:
        raise SystemExit("Provide video paths or --demo")

    out = build_gif(
        vids,
        args.output,
        cell_w=args.cell_w,
        cell_h=args.cell_h,
        gap=args.gap,
        duration=args.duration,
        fps=args.fps,
    )
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")

    still = args.still
    if still is None:
        still = out.with_suffix(".png")
    # Extract first frame as RGBA PNG for review (keep transparency).
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(out),
            "-vframes",
            "1",
            "-pix_fmt",
            "rgba",
            str(still),
        ],
        check=True,
    )
    print(f"Still frame: {still}")


if __name__ == "__main__":
    main()
