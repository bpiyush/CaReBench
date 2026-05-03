"""
Synthetic Spatial-Motion Video Dataset Generator
=================================================
Renders short MP4 clips of a single moving object against a background.
Each clip is saved alongside a JSON metadata file with all configuration
details, ground-truth direction label, speed label, and natural-language
query strings.

Randomised axes
---------------
- Object shape      : circle, square, rectangle, triangle, diamond
- Object size       : small / medium / large (relative to frame)
- Object            : solid fill (single random RGB + black outline)
- Background texture: solid or gradient only
- Background color(s): random RGB (distinct from object)
- Direction class   : up / down / left / right
- Speed category    : slow / medium / fast
- Velocity (px/frame): sampled within speed-band, with slight jitter
- Frame count       : 48 – 72 frames (randomised)
- FPS               : 24

Encoding
--------
Videos are written with **ffmpeg** (H.264 / yuv420p), not OpenCV's
``VideoWriter``, so previews in Cursor and typical players work reliably.
``ffmpeg`` must be on ``PATH`` (install the ``ffmpeg`` package if encoding fails).

Usage
-----
    python generate_spatial_dataset.py [--output_dir DATASET] \
                                       [--n_per_class N]     \
                                       [--seed SEED]          \
                                       [--width W] [--height H]

Produces
--------
    DATASET/
        videos/
            0000.mp4
            0001.mp4
            ...
        metadata/
            0000.json
            0001.json
            ...
        index.json          <- summary of the whole dataset
"""

import argparse
import json
import os
import random
import shutil
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Constants / vocabulary
# ---------------------------------------------------------------------------

DIRECTIONS = ["up", "down", "left", "right"]

SPEED_BANDS = {
    "slow":   (1, 2),    # px / frame
    "medium": (3, 5),
    "fast":   (6, 10),
}

SHAPES = ["circle", "square", "rectangle", "triangle", "diamond"]

OBJECT_TEXTURES = ["solid"]  # uniform fill + black border (see build_object_patch)

BACKGROUND_TEXTURES = ["solid", "gradient"]

# Outline width when drawing the object border (pixels, scaled with object size).
OBJECT_BORDER_THICKNESS_RATIO = 0.04
OBJECT_BORDER_THICKNESS_MIN = 2
OBJECT_BORDER_THICKNESS_MAX = 4

# Natural-language query templates  (direction, speed, shape are substituted)
QUERY_TEMPLATES = [
    "{shape} moving {direction}",
    "{shape} moving {direction} {speed}",
    "{shape} going {direction}",
    "{speed} {shape} moving {direction}",
    "{shape} drifting {direction} {speed}",
    "object moving {direction} at {speed} speed",
    "{shape} sliding {direction}",
    "{shape} traveling {direction} {speed}",
]

# Direction -> (dx, dy) per frame
DIRECTION_VECTORS = {
    "up":    ( 0, -1),
    "down":  ( 0,  1),
    "left":  (-1,  0),
    "right": ( 1,  0),
}


# ---------------------------------------------------------------------------
# Texture helpers
# ---------------------------------------------------------------------------

def make_solid_patch(h: int, w: int, color: Tuple[int, int, int]) -> np.ndarray:
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[:] = color
    return patch


def make_gradient_patch(h: int, w: int,
                         c1: Tuple[int, int, int],
                         c2: Tuple[int, int, int],
                         direction: str = "horizontal") -> np.ndarray:
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    if direction == "horizontal":
        for x in range(w):
            t = x / max(w - 1, 1)
            patch[:, x] = [int(c1[k] * (1 - t) + c2[k] * t) for k in range(3)]
    else:
        for y in range(h):
            t = y / max(h - 1, 1)
            patch[y, :] = [int(c1[k] * (1 - t) + c2[k] * t) for k in range(3)]
    return patch


def make_checkerboard_patch(h: int, w: int,
                             c1: Tuple[int, int, int],
                             c2: Tuple[int, int, int],
                             cell: int = 16) -> np.ndarray:
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if (y // cell + x // cell) % 2 == 0:
                patch[y, x] = c1
            else:
                patch[y, x] = c2
    return patch


def make_stripes_patch(h: int, w: int,
                        c1: Tuple[int, int, int],
                        c2: Tuple[int, int, int],
                        stripe_w: int = 12,
                        horizontal: bool = False) -> np.ndarray:
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    if horizontal:
        for y in range(h):
            patch[y, :] = c1 if (y // stripe_w) % 2 == 0 else c2
    else:
        for x in range(w):
            patch[:, x] = c1 if (x // stripe_w) % 2 == 0 else c2
    return patch


def make_noise_patch(h: int, w: int,
                      c1: Tuple[int, int, int],
                      c2: Tuple[int, int, int],
                      rng: np.random.Generator,
                      scale: int = 8) -> np.ndarray:
    """Block noise: low-frequency random color patches."""
    bh = max(h // scale, 1)
    bw = max(w // scale, 1)
    blocks = rng.integers(0, 2, (bh, bw), dtype=np.uint8)
    # Upscale
    big = cv2.resize(blocks, (w, h), interpolation=cv2.INTER_NEAREST)
    patch = np.where(big[:, :, None] == 0,
                     np.array(c1, dtype=np.uint8),
                     np.array(c2, dtype=np.uint8))
    return patch.astype(np.uint8)


def build_background(h: int, w: int, texture: str,
                      c1: Tuple[int, int, int],
                      c2: Tuple[int, int, int],
                      rng: np.random.Generator) -> np.ndarray:
    if texture == "solid":
        return make_solid_patch(h, w, c1)
    elif texture == "gradient":
        d = rng.choice(["horizontal", "vertical"])
        return make_gradient_patch(h, w, c1, c2, direction=d)
    elif texture == "checkerboard":
        cell = int(rng.integers(8, 28))
        return make_checkerboard_patch(h, w, c1, c2, cell=cell)
    elif texture == "horizontal_stripes":
        sw = int(rng.integers(6, 20))
        return make_stripes_patch(h, w, c1, c2, stripe_w=sw, horizontal=True)
    elif texture == "vertical_stripes":
        sw = int(rng.integers(6, 20))
        return make_stripes_patch(h, w, c1, c2, stripe_w=sw, horizontal=False)
    elif texture == "noise":
        return make_noise_patch(h, w, c1, c2, rng)
    else:
        return make_solid_patch(h, w, c1)


# ---------------------------------------------------------------------------
# Object mask helpers
# ---------------------------------------------------------------------------

def draw_shape_mask(shape: str, size: int) -> np.ndarray:
    """Return a (size, size) uint8 mask with the shape filled as 255."""
    mask = np.zeros((size, size), dtype=np.uint8)
    c = size // 2
    r = c - 2  # slight inset so edges are clean

    if shape == "circle":
        cv2.circle(mask, (c, c), r, 255, -1)

    elif shape == "square":
        m = size // 6
        cv2.rectangle(mask, (m, m), (size - m, size - m), 255, -1)

    elif shape == "rectangle":
        # wider than tall
        pw = int(size * 0.80)
        ph = int(size * 0.50)
        x0 = (size - pw) // 2
        y0 = (size - ph) // 2
        cv2.rectangle(mask, (x0, y0), (x0 + pw, y0 + ph), 255, -1)

    elif shape == "triangle":
        pts = np.array([
            [c, 2],
            [2, size - 3],
            [size - 3, size - 3],
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)

    elif shape == "diamond":
        pts = np.array([
            [c, 2],
            [size - 3, c],
            [c, size - 3],
            [2, c],
        ], np.int32)
        cv2.fillPoly(mask, [pts], 255)

    return mask


def build_object_patch(shape: str, size: int, _texture: str,
                        c1: Tuple[int, int, int],
                        _c2: Tuple[int, int, int],
                        _rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (rgb_patch [size,size,3], alpha_mask [size,size]) both uint8.

    Object is always a uniform fill (c1) with a black outline; texture/c2
    are accepted for call-site compatibility but not used for appearance.
    """
    mask = draw_shape_mask(shape, size)
    patch = make_solid_patch(size, size, c1)

    thick = int(round(size * OBJECT_BORDER_THICKNESS_RATIO))
    thick = max(OBJECT_BORDER_THICKNESS_MIN, min(OBJECT_BORDER_THICKNESS_MAX, thick))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(patch, contours, -1, (0, 0, 0), thickness=thick)

    alpha = mask
    return patch, alpha


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------

def composite(background: np.ndarray,
               obj_patch: np.ndarray,
               obj_alpha: np.ndarray,
               cx: int, cy: int,
               obj_size: int) -> np.ndarray:
    """
    Paste obj_patch onto background centred at (cx, cy).
    Handles partial out-of-frame clipping.
    """
    frame = background.copy()
    H, W = frame.shape[:2]
    half = obj_size // 2

    # Object occupies [cx-half, cx-half+obj_size) in frame coords
    obj_x0 = cx - half
    obj_y0 = cy - half

    # Destination region (clipped to frame)
    dx0 = max(0, obj_x0)
    dy0 = max(0, obj_y0)
    dx1 = min(W, obj_x0 + obj_size)
    dy1 = min(H, obj_y0 + obj_size)

    if dx1 <= dx0 or dy1 <= dy0:
        return frame  # fully off-screen

    # Corresponding source region
    sx0 = dx0 - obj_x0
    sy0 = dy0 - obj_y0
    sx1 = sx0 + (dx1 - dx0)
    sy1 = sy0 + (dy1 - dy0)

    src_patch = obj_patch[sy0:sy1, sx0:sx1]
    src_alpha = obj_alpha[sy0:sy1, sx0:sx1].astype(np.float32) / 255.0

    dst = frame[dy0:dy1, dx0:dx1].astype(np.float32)
    blended = src_patch.astype(np.float32) * src_alpha[:, :, None] + \
              dst * (1.0 - src_alpha[:, :, None])
    frame[dy0:dy1, dx0:dx1] = blended.clip(0, 255).astype(np.uint8)
    return frame


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def random_color(rng: np.random.Generator,
                 exclude: Tuple[int, int, int] = None,
                 min_dist: int = 80) -> Tuple[int, int, int]:
    """Sample a random bright-ish color, optionally far from `exclude`."""
    for _ in range(50):
        c = tuple(int(x) for x in rng.integers(30, 240, 3))
        if exclude is None:
            return c
        dist = sum(abs(c[k] - exclude[k]) for k in range(3))
        if dist >= min_dist:
            return c
    return tuple(int(x) for x in rng.integers(30, 240, 3))


COLOR_NAMES = {
    # simple nearest-name lookup
    "red": (220, 50, 50), "green": (50, 200, 50), "blue": (50, 50, 220),
    "yellow": (220, 220, 50), "cyan": (50, 220, 220), "magenta": (220, 50, 220),
    "orange": (220, 130, 50), "purple": (130, 50, 220), "white": (230, 230, 230),
    "gray": (128, 128, 128),
}

def nearest_color_name(c: Tuple[int, int, int]) -> str:
    best, best_d = "colorful", 1e9
    for name, ref in COLOR_NAMES.items():
        d = sum((c[k] - ref[k]) ** 2 for k in range(3))
        if d < best_d:
            best_d, best = d, name
    return best


# ---------------------------------------------------------------------------
# Query generation
# ---------------------------------------------------------------------------

def make_queries(shape: str, direction: str, speed: str, color_name: str) -> List[str]:
    queries = set()
    for tmpl in QUERY_TEMPLATES:
        q = tmpl.format(shape=shape, direction=direction, speed=speed,
                        color=color_name)
        queries.add(q)
    # A few color-aware extras
    extras = [
        f"{color_name} {shape} moving {direction}",
        f"{color_name} {shape} going {direction} {speed}",
        f"{shape} with {color_name} color moving {direction}",
    ]
    queries.update(extras)
    return sorted(queries)


# ---------------------------------------------------------------------------
# Single video generation
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    video_id: str
    seed: int
    width: int
    height: int
    fps: int
    n_frames: int
    direction: str
    speed_label: str
    velocity_px_per_frame: float
    shape: str
    obj_size_label: str
    obj_size_px: int
    obj_texture: str
    obj_color_primary: Tuple[int, int, int]
    obj_color_secondary: Tuple[int, int, int]
    obj_color_name: str
    bg_texture: str
    bg_color_primary: Tuple[int, int, int]
    bg_color_secondary: Tuple[int, int, int]
    start_cx: int
    start_cy: int
    queries: List[str] = field(default_factory=list)
    label: str = ""          # direction class
    sub_label: str = ""      # direction + speed


def sample_config(video_id: str, direction: str, seed: int,
                  width: int, height: int) -> VideoConfig:
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    # --- timing ---
    fps = 24
    n_frames = int(rng.integers(48, 73))  # 2 – 3 seconds

    # --- speed ---
    speed_label = py_rng.choice(list(SPEED_BANDS.keys()))
    lo, hi = SPEED_BANDS[speed_label]
    velocity = float(rng.uniform(lo, hi + 0.5))

    # --- shape & size ---
    shape = py_rng.choice(SHAPES)
    size_label = py_rng.choice(["small", "medium", "large"])
    size_map = {
        "small":  int(min(width, height) * 0.12),
        "medium": int(min(width, height) * 0.22),
        "large":  int(min(width, height) * 0.32),
    }
    obj_size = size_map[size_label]

    # --- colors ---
    bg_c1 = random_color(rng)
    bg_c2 = random_color(rng, exclude=bg_c1, min_dist=60)
    obj_c1 = random_color(rng, exclude=bg_c1, min_dist=80)
    obj_c2 = obj_c1  # uniform object color (metadata matches render)
    color_name = nearest_color_name(obj_c1)

    # --- textures ---
    obj_tex = "solid"
    bg_tex = py_rng.choice(BACKGROUND_TEXTURES)

    # --- trajectory ---
    # Centre the motion path so the midpoint is at frame centre
    dx, dy = DIRECTION_VECTORS[direction]
    total_travel = velocity * n_frames
    half_travel = total_travel / 2
    start_cx = int(width // 2 - dx * half_travel)
    start_cy = int(height // 2 - dy * half_travel)
    # Clamp so object starts at least partially visible
    half = obj_size // 2
    start_cx = int(np.clip(start_cx, -half, width + half))
    start_cy = int(np.clip(start_cy, -half, height + half))

    queries = make_queries(shape, direction, speed_label, color_name)

    return VideoConfig(
        video_id=video_id,
        seed=seed,
        width=width,
        height=height,
        fps=fps,
        n_frames=n_frames,
        direction=direction,
        speed_label=speed_label,
        velocity_px_per_frame=round(velocity, 3),
        shape=shape,
        obj_size_label=size_label,
        obj_size_px=obj_size,
        obj_texture=obj_tex,
        obj_color_primary=obj_c1,
        obj_color_secondary=obj_c2,
        obj_color_name=color_name,
        bg_texture=bg_tex,
        bg_color_primary=bg_c1,
        bg_color_secondary=bg_c2,
        start_cx=start_cx,
        start_cy=start_cy,
        queries=queries,
        label=direction,
        sub_label=f"{direction}_{speed_label}",
    )


def _write_mp4_via_ffmpeg(
    video_path: str,
    width: int,
    height: int,
    fps: int,
    n_frames: int,
    frame_gen,
) -> None:
    """
    Encode RGB uint8 frames (H, W, 3) to MP4 using ffmpeg stdin.
    Uses H.264 + yuv420p for broad preview/player compatibility.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; install ffmpeg to write videos "
            "(e.g. apt install ffmpeg / brew install ffmpeg)."
        )

    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        video_path,
    ]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    assert proc.stdin is not None
    try:
        for _ in range(n_frames):
            frame = next(frame_gen)
            if frame.shape != (height, width, 3) or frame.dtype != np.uint8:
                raise ValueError(
                    f"Expected frame ({height}, {width}, 3) uint8 RGB, "
                    f"got {frame.shape} {frame.dtype}"
                )
            if not frame.flags["C_CONTIGUOUS"]:
                frame = np.ascontiguousarray(frame)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
        proc.wait(timeout=300)
        stderr = proc.stderr.read() if proc.stderr else b""
        if proc.returncode != 0:
            msg = stderr.decode(errors="replace") if stderr else "(no stderr)"
            raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {msg}")
    except BaseException:
        proc.stdin.close()
        proc.kill()
        proc.wait(timeout=30)
        raise


def render_video(cfg: VideoConfig, video_path: str) -> None:
    rng = np.random.default_rng(cfg.seed + 1)

    # Build static background once
    bg = build_background(cfg.height, cfg.width, cfg.bg_texture,
                          cfg.bg_color_primary, cfg.bg_color_secondary, rng)

    # Build object patch once (it translates rigidly)
    obj_patch, obj_alpha = build_object_patch(
        cfg.shape, cfg.obj_size_px, cfg.obj_texture,
        cfg.obj_color_primary, cfg.obj_color_secondary, rng
    )

    dx, dy = DIRECTION_VECTORS[cfg.direction]

    def frames():
        for t in range(cfg.n_frames):
            cx = int(round(cfg.start_cx + dx * cfg.velocity_px_per_frame * t))
            cy = int(round(cfg.start_cy + dy * cfg.velocity_px_per_frame * t))
            yield composite(bg, obj_patch, obj_alpha, cx, cy, cfg.obj_size_px)

    _write_mp4_via_ffmpeg(
        video_path,
        cfg.width,
        cfg.height,
        cfg.fps,
        cfg.n_frames,
        frames(),
    )


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(output_dir: str,
                     n_per_class: int = 25,
                     seed: int = 42,
                     width: int = 256,
                     height: int = 256) -> None:
    video_dir = os.path.join(output_dir, "videos")
    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    rng = random.Random(seed)
    all_meta = []
    vid_idx = 0

    for direction in DIRECTIONS:
        for _ in range(n_per_class):
            video_id = f"{vid_idx:04d}"
            clip_seed = rng.randint(0, 2**31)

            cfg = sample_config(video_id, direction, clip_seed, width, height)

            video_path = os.path.join(video_dir, f"{video_id}.mp4")
            meta_path = os.path.join(meta_dir, f"{video_id}.json")

            render_video(cfg, video_path)

            # Serialise config (convert tuples to lists for JSON)
            meta = asdict(cfg)
            meta["video_file"] = f"videos/{video_id}.mp4"
            meta["obj_color_primary"] = list(meta["obj_color_primary"])
            meta["obj_color_secondary"] = list(meta["obj_color_secondary"])
            meta["bg_color_primary"] = list(meta["bg_color_primary"])
            meta["bg_color_secondary"] = list(meta["bg_color_secondary"])

            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            all_meta.append({
                "video_id": video_id,
                "video_file": meta["video_file"],
                "label": direction,
                "sub_label": cfg.sub_label,
                "shape": cfg.shape,
                "speed_label": cfg.speed_label,
                "velocity_px_per_frame": cfg.velocity_px_per_frame,
                "obj_texture": cfg.obj_texture,
                "bg_texture": cfg.bg_texture,
                "n_frames": cfg.n_frames,
                "queries": cfg.queries,
            })

            vid_idx += 1
            print(f"  [{vid_idx:>4}/{len(DIRECTIONS)*n_per_class}] "
                  f"{video_id}.mp4  dir={direction:5s}  "
                  f"speed={cfg.speed_label:6s}  shape={cfg.shape:9s}  "
                  f"obj_tex={cfg.obj_texture:12s}  bg_tex={cfg.bg_texture}")

    # Write index
    index = {
        "total_videos": len(all_meta),
        "n_per_class": n_per_class,
        "classes": DIRECTIONS,
        "speed_labels": list(SPEED_BANDS.keys()),
        "shapes": SHAPES,
        "object_textures": OBJECT_TEXTURES,
        "background_textures": BACKGROUND_TEXTURES,
        "frame_size": [width, height],
        "fps": 24,
        "global_seed": seed,
        "videos": all_meta,
    }
    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. {len(all_meta)} videos written to '{output_dir}/'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic spatial-motion video dataset"
    )
    parser.add_argument("--output_dir", default="/scratch/shared/beegfs/piyush/datasets/SyntheticMotion/spatial_motion_dataset",
                        help="Root directory for the dataset (default: spatial_motion_dataset)")
    parser.add_argument("--n_per_class", type=int, default=25,
                        help="Number of videos per direction class (default: 25)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global random seed (default: 42)")
    parser.add_argument("--width", type=int, default=256,
                        help="Frame width in pixels (default: 256)")
    parser.add_argument("--height", type=int, default=256,
                        help="Frame height in pixels (default: 256)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing dataset")
    args = parser.parse_args()
    
    if args.overwrite:
        print(f"Overwriting existing dataset at {args.output_dir}.")
        import shutil
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir)

    print(f"Generating {args.n_per_class * len(DIRECTIONS)} videos "
          f"({args.n_per_class} per class × {len(DIRECTIONS)} classes) "
          f"into '{args.output_dir}/'")
    print(f"Frame size: {args.width}×{args.height}  |  Seed: {args.seed}\n")

    generate_dataset(
        output_dir=args.output_dir,
        n_per_class=args.n_per_class,
        seed=args.seed,
        width=args.width,
        height=args.height,
    )