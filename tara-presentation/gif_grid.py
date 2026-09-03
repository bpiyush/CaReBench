"""Build a 3×3 video grid GIF with transparent equal gaps (no labels/cards)."""
from __future__ import annotations

import subprocess
from pathlib import Path


def build_gif(
    videos: list[Path],
    out_gif: Path,
    *,
    cell_w: int = 320,
    cell_h: int = 200,
    gap: int = 8,
    duration: float = 4.0,
    fps: int = 10,
) -> Path:
    if not (1 <= len(videos) <= 9):
        raise ValueError("Need 1–9 videos")
    videos = list(videos)[:9]
    while len(videos) < 9:
        videos.append(videos[-1])

    out_gif.parent.mkdir(parents=True, exist_ok=True)

    grid_w = 3 * cell_w + 2 * gap
    grid_h = 3 * cell_h + 2 * gap
    step_x, step_y = cell_w + gap, cell_h + gap

    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    # Transparent canvas — gaps stay alpha; no chromakey (avoids eating pink/magenta video).
    cmd += [
        "-f",
        "lavfi",
        "-i",
        f"color=c=black@0.0:s={grid_w}x{grid_h}:d={duration:.3f}:r={fps},format=rgba",
    ]
    filters: list[str] = []

    for i, path in enumerate(videos):
        cmd += ["-stream_loop", "-1", "-t", f"{duration:.3f}", "-i", str(path)]
        filters.append(
            f"[{i + 1}:v]fps={fps},"
            f"scale={cell_w}:{cell_h}:force_original_aspect_ratio=increase,"
            f"crop={cell_w}:{cell_h},"
            f"setsar=1,format=rgba[v{i}]"
        )

    # Overlay tiles onto the transparent canvas at equal horizontal/vertical gaps.
    prev = "[0:v]"
    for i in range(9):
        col, row = i % 3, i // 3
        x, y = col * step_x, row * step_y
        out_lbl = f"[o{i}]" if i < 8 else "[rgba]"
        filters.append(f"{prev}[v{i}]overlay={x}:{y}:format=auto{out_lbl}")
        prev = out_lbl

    # Palette with reserved transparent index.
    filters.append(
        "[rgba]split[s0][s1];"
        "[s0]palettegen=max_colors=256:reserve_transparent=1:stats_mode=single[p];"
        "[s1][p]paletteuse=dither=bayer:bayer_scale=3:alpha_threshold=128[out]"
    )

    cmd += [
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[out]",
        "-t",
        f"{duration:.3f}",
        "-loop",
        "0",
        str(out_gif),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("ffmpeg timed out") from exc
    if proc.returncode != 0 or not out_gif.exists() or out_gif.stat().st_size == 0:
        err = (proc.stderr or proc.stdout or "").strip()[-2000:]
        raise RuntimeError(f"ffmpeg failed:\n{err}")
    return out_gif
