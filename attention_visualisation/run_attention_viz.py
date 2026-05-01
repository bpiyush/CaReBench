"""
run_attention_viz.py
====================
Command-line runner that loads an MLLM, runs one forward pass over a video,
identifies attention sinks, suppresses them, and saves a side-by-side PNG of
the raw frames and the attention-heatmap overlays.

Usage
-----
python run_attention_viz.py \\
    --video       /path/to/video.mp4 \\
    --model_path  /path/to/TARA \\
    --out         /path/to/output.png \\
    [--model_name TARA] \\
    [--nframes 4] \\
    [--layer -1] \\
    [--sink_tau 20] \\
    [--suppress_sinks] \\
    [--blend_alpha 0.3] \\
    [--attn_implementation eager] \\
    [--dtype bfloat16]
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ── project imports ──────────────────────────────────────────────────────────
from models.modeling_encoders import AutoEncoder
import shared.utils as su
from attention_visualisation.compute_video_attentions import (
    align_attn_to_frames,
    build_model_inputs,
    build_visual_index_map,
    dtype_from_name,
    frames_tchw_to_pil_list,
    resolve_prompt,
    run_embedding_and_attention,
    unpatchify_qwen2vl_pixel_values,
)

# ── sink utilities ────────────────────────────────────────────────────────────

# Default sink feature dimensions identified for TARA / Tarsier2-7b.
# Override via --sink_dims if your checkpoint differs.
DEFAULT_SINK_DIMS = [458, 2570]


def compute_sink_score(
    x: torch.Tensor,
    sink_dims: list[int],
) -> torch.Tensor:
    """
    Normalised activation magnitude at known sink dimensions.

    x : [..., D]  (any leading dims)
    Returns [...] scalar score per token / layer.
    """
    if x.ndim == 1:
        x = x.unsqueeze(0)
    norm = torch.linalg.norm(x, dim=-1, keepdim=True) / (x.shape[-1] ** 0.5)
    # Guard against zero-norm vectors
    norm = norm.clamp(min=1e-8)
    return (x[..., sink_dims] / norm).abs().max(dim=-1).values


def find_visual_sink_tokens(
    hidden_states_lsd: torch.Tensor,   # [L, S, D]
    visual_start: int,
    visual_end_exclusive: int,
    sink_dims: list[int],
    tau: float,
    hidden_dim: int,
) -> torch.Tensor:
    """
    Return the indices (within the visual block) of tokens whose mean
    sink score across layers exceeds tau.
    """
    max_dim = hidden_states_lsd.shape[-1]
    bad = [d for d in sink_dims if d >= max_dim]
    if bad:
        raise ValueError(
            f"sink_dims {bad} are out of range for hidden_dim={max_dim}. "
            "Pass --sink_dims with valid indices for your model."
        )

    import einops

    ls = einops.rearrange(
        compute_sink_score(
            einops.rearrange(hidden_states_lsd, "l s d -> (l s) d"),
            sink_dims=sink_dims,
        ),
        "(l s) -> l s",
        l=len(hidden_states_lsd),
    )                                          # [L, S]

    ls_v = ls[:, visual_start:visual_end_exclusive]  # [L, N_vis]
    mean_score = ls_v.mean(dim=0)             # [N_vis]
    return torch.where(mean_score >= tau)[0]


# ── image helpers ─────────────────────────────────────────────────────────────

def tensor_to_heatmap_pil(
    attn: torch.Tensor,   # [H, W]  float
    cmap: str = "jet",
) -> Image.Image:
    a = attn.float().numpy()
    a_min, a_max = a.min(), a.max()
    if a_max > a_min:
        a = (a - a_min) / (a_max - a_min)
    colormap = cm.get_cmap(cmap)
    rgba = colormap(a)                           # [H, W, 4]  float64
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def blend_pil(base: Image.Image, overlay: Image.Image, alpha: float) -> Image.Image:
    """Alpha-blend overlay onto base.  Both must be the same size."""
    if base.size != overlay.size:
        overlay = overlay.resize(base.size, Image.BILINEAR)
    return Image.blend(base, overlay, alpha)


def add_label(img: Image.Image, text: str, font_size: int = 18) -> Image.Image:
    """Paste a text label onto a copy of img."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.rectangle([0, 0, img.width, font_size + 6], fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return out


def make_grid(images: list[Image.Image], ncols: int, gap: int = 4) -> Image.Image:
    """Arrange PIL images into a grid with a black gap between cells."""
    nrows = (len(images) + ncols - 1) // ncols
    w, h = images[0].size
    grid_w = ncols * w + (ncols - 1) * gap
    grid_h = nrows * h + (nrows - 1) * gap
    grid = Image.new("RGB", (grid_w, grid_h), color=(0, 0, 0))
    for idx, img in enumerate(images):
        col = idx % ncols
        row = idx // ncols
        x = col * (w + gap)
        y = row * (h + gap)
        grid.paste(img.resize((w, h), Image.BILINEAR), (x, y))
    return grid


def save_figure(
    pil_frames: list[Image.Image],
    heatmap_frames: list[Image.Image],
    out_path: str,
    title: str,
    ncols: int,
) -> None:
    """
    Save a two-row PNG:
      Row 1 – raw decoded frames
      Row 2 – heatmap-overlaid frames
    """
    labeled_raw = [add_label(f, f"Frame {i}") for i, f in enumerate(pil_frames)]
    labeled_heat = [add_label(f, f"Attn {i}") for i, f in enumerate(heatmap_frames)]

    top = make_grid(labeled_raw, ncols=ncols)
    bot = make_grid(labeled_heat, ncols=ncols)

    # Stack rows with a thin separator
    sep = 8
    canvas_w = max(top.width, bot.width)
    canvas_h = top.height + sep + bot.height + 40  # 40px for title bar
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(30, 30, 30))

    canvas.paste(top, (0, 40))
    canvas.paste(bot, (0, 40 + top.height + sep))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20
        )
    except (IOError, OSError):
        font = ImageFont.load_default()
    draw.text((10, 8), title, fill=(255, 255, 255), font=font)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    canvas.save(out_path, format="PNG")
    print(f"Saved → {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualise MLLM video attention maps.")
    p.add_argument("--video", required=True, help="Path to input video file.")
    p.add_argument("--model_path", required=True, help="Path to pretrained model directory.")
    p.add_argument("--out", default="attention_viz.png", help="Output PNG path.")
    p.add_argument("--model_name", default="MLLM", help="Display name for the model.")
    p.add_argument("--base_config", default="nframes=4.yaml",
                   help="Base config name passed to AutoEncoder.")
    p.add_argument("--layer", type=int, default=-1,
                   help="Attention layer to visualise (-1 = last, None = mean all). "
                        "Pass --layer=-1 for last layer or omit for mean.")
    p.add_argument("--mean_layers", action="store_true",
                   help="If set, average attention across all layers instead of picking one.")
    p.add_argument("--sink_tau", type=float, default=20.0,
                   help="Sink-score threshold τ for identifying visual sink tokens.")
    p.add_argument("--sink_dims", type=int, nargs="+", default=DEFAULT_SINK_DIMS,
                   help="Feature dimensions used to detect attention sinks.")
    p.add_argument("--suppress_sinks", action="store_true", default=True,
                   help="Zero out visual sink tokens before computing heatmaps.")
    p.add_argument("--no_suppress_sinks", dest="suppress_sinks", action="store_false")
    p.add_argument("--blend_alpha", type=float, default=0.35,
                   help="Heatmap blend alpha (0 = raw frame, 1 = pure heatmap).")
    p.add_argument("--cmap", default="jet", help="Matplotlib colormap for heatmaps.")
    p.add_argument("--attn_implementation", default="eager",
                   choices=["eager", "sdpa", "flash_attention_2"],
                   help="Attention implementation (must support weight output).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "float16", "bfloat16"],
                   help="Model dtype.")
    p.add_argument("--ncols", type=int, default=None,
                   help="Number of columns in the output grid (default: number of frames).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── validate inputs ───────────────────────────────────────────────────────
    if not os.path.exists(args.video):
        sys.exit(f"[ERROR] Video not found: {args.video}")
    if not os.path.isdir(args.model_path):
        sys.exit(f"[ERROR] Model path not found: {args.model_path}")

    model_dtype = dtype_from_name(args.dtype)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"Loading model from {args.model_path} …")
    encoder = AutoEncoder.from_pretrained(
        args.model_path,
        device_map="auto",
        attn_implementation=args.attn_implementation,
        dtype=model_dtype,
        base_config_name=args.base_config,
    )
    encoder.model.eval()
    print("Model loaded.")

    hidden_dim = encoder.model.language_model.config.hidden_size

    # ── prepare inputs ────────────────────────────────────────────────────────
    prompt = resolve_prompt(encoder, args.video, custom_prompt=None)
    model_inputs = build_model_inputs(encoder, args.video, prompt)
    m = build_visual_index_map(model_inputs, encoder)  # raises on span mismatch (CRITICAL-2)

    S = model_inputs["input_ids"].shape[-1]
    print(f"Sequence length: {S}")
    print(f"Visual tokens: {m['N_vis']}  (T={m['T_llm']}, H={m['H_llm']}, W={m['W_llm']})")

    # ── forward pass ──────────────────────────────────────────────────────────
    print("Running forward pass …")
    embedding, attention_lhss, hidden_states_lsd = run_embedding_and_attention(
        encoder,
        model_inputs,
        save_dtype=torch.float32,
    )
    print(f"  embedding:       {tuple(embedding.shape)}")
    print(f"  attention_lhss:  {tuple(attention_lhss.shape)}")
    print(f"  hidden_states:   {tuple(hidden_states_lsd.shape)}")

    # ── identify sink tokens ──────────────────────────────────────────────────
    visual_sink_tokens = find_visual_sink_tokens(
        hidden_states_lsd,
        visual_start=m["visual_start"],
        visual_end_exclusive=m["visual_end_exclusive"],
        sink_dims=args.sink_dims,
        tau=args.sink_tau,
        hidden_dim=hidden_dim,
    )
    print(f"Visual sink tokens found: {len(visual_sink_tokens)}")

    # ── extract and reshape visual attention ──────────────────────────────────
    vs, ve = m["visual_start"], m["visual_end_exclusive"]
    T_llm, H_llm, W_llm = m["T_llm"], m["H_llm"], m["W_llm"]

    # attention from the last generated token back to each visual token
    a_vis = attention_lhss[:, :, -1, vs:ve].clone()   # [L, H, N_vis]

    if args.suppress_sinks and len(visual_sink_tokens) > 0:
        print("Suppressing attention sinks …")
        a_vis[:, :, visual_sink_tokens] = 0.0

    # [L, H, N_vis] → [L, H, T, H_p, W_p]
    a_vis_thw = a_vis.reshape(
        a_vis.shape[0], a_vis.shape[1], T_llm, H_llm, W_llm
    )

    # ── decode frames from pixel_values ──────────────────────────────────────
    ip = encoder.processor.image_processor
    frames_tchw = unpatchify_qwen2vl_pixel_values(
        pixel_values=model_inputs["pixel_values"],
        image_grid_thw=model_inputs["image_grid_thw"][0],
        C=3,
        temporal_patch_size=ip.temporal_patch_size,
        patch_size=ip.patch_size,
        merge_size=ip.merge_size,
    )
    pil_frames = frames_tchw_to_pil_list(frames_tchw)
    print(f"Decoded {len(pil_frames)} frames at {pil_frames[0].size} px")

    # ── align attention to frames ─────────────────────────────────────────────
    # CRITICAL-1 is fixed: layer_idx / head_idx are the parameter names now.
    layer_idx: int | None = None if args.mean_layers else args.layer

    attn_frame_hw = align_attn_to_frames(
        a_vis_thw,
        frames_tchw,
        temporal_patch_size=ip.temporal_patch_size,
        layer_idx=layer_idx,
        head_idx=None,   # always mean over heads
    )
    print(f"Attention map shape: {tuple(attn_frame_hw.shape)}")  # [T_img, H', W']

    # ── build overlay images ──────────────────────────────────────────────────
    heatmap_frames = []
    for i, (frame_pil, attn_hw) in enumerate(zip(pil_frames, attn_frame_hw)):
        hm = tensor_to_heatmap_pil(attn_hw, cmap=args.cmap)
        overlay = blend_pil(frame_pil, hm, alpha=args.blend_alpha)
        heatmap_frames.append(overlay)

    # ── save PNG ──────────────────────────────────────────────────────────────
    ncols = args.ncols if args.ncols is not None else len(pil_frames)
    sink_tag = "no sinks" if args.suppress_sinks else "with sinks"
    layer_tag = "mean layers" if layer_idx is None else f"layer {layer_idx}"
    title = f"{args.model_name} — attention heatmaps ({sink_tag}, {layer_tag})"

    save_figure(
        pil_frames=pil_frames,
        heatmap_frames=heatmap_frames,
        out_path=args.out,
        title=title,
        ncols=ncols,
    )


if __name__ == "__main__":
    main()