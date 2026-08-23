"""Generate lightweight but smooth preview clips for UI playback."""
from __future__ import annotations

import subprocess
from pathlib import Path

from config import PREVIEW_WIDTH, PREVIEWS_DIR


def preview_cache_dir(cache_key: str) -> Path:
    out = PREVIEWS_DIR / cache_key
    out.mkdir(parents=True, exist_ok=True)
    return out


def _is_bad_preview(path: str | Path) -> bool:
    """Reject ultra-low-FPS feature extracts (look like slideshows)."""
    return "msrvtt_2fps_224" in str(path)


def get_or_create_preview(
    source_path: str,
    cache_key: str,
    video_id: str,
    existing_preview: str | None = None,
) -> str:
    """Downscale for fast UI playback while keeping the source frame rate."""
    out_dir = preview_cache_dir(cache_key)
    safe_id = video_id.replace("/", "_")
    out_path = out_dir / f"{safe_id}.mp4"
    if out_path.exists():
        return str(out_path)

    # Prefer original source; never use the 2fps MSRVTT extract for display.
    src = Path(source_path)
    if not src.exists() and existing_preview and not _is_bad_preview(existing_preview):
        src = Path(existing_preview)
    if not src.exists():
        return source_path

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        f"scale={PREVIEW_WIDTH}:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "26",
        "-movflags",
        "+faststart",
        "-an",
        str(out_path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=180)
        return str(out_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return str(src)
