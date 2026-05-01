import argparse
import os
from typing import Dict, List, Tuple

import torch
import matplotlib.pyplot as plt

from models.modeling_encoders import AutoEncoder
from models.tarsier2.dataset.utils import format_one_sample
from typing import List, Sequence, Tuple
import torch
import torch.nn.functional as F
from PIL import Image

import shared.utils as su

VIDEO_EXTENSIONS = {"mp4", "avi", "mov", "mkv", "webm"}


def infer_is_video(media_path: str) -> bool:
    ext = media_path.rsplit(".", 1)[-1].lower()
    return ext in VIDEO_EXTENSIONS


def resolve_prompt(encoder, media_path: str, custom_prompt: str | None = None) -> str:
    if custom_prompt is not None:
        if "<video>" not in custom_prompt and "<image>" not in custom_prompt:
            raise ValueError("Custom prompt must contain <video> or <image>.")
        return custom_prompt
    return encoder.video_eol_prompt if infer_is_video(media_path) else encoder.image_eol_prompt


def build_model_inputs(encoder, media_path: str, prompt: str) -> Dict[str, torch.Tensor]:
    sample = format_one_sample(media_file=media_path, prompt=prompt)
    sample = encoder.super_processor(sample)

    model_inputs: Dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            model_inputs[key] = value.to(encoder.model.device)
    return model_inputs


def _flatten_first_generation_step_attentions(attentions) -> List[torch.Tensor]:
    current = attentions
    while isinstance(current, (list, tuple)) and len(current) > 0 and isinstance(current[0], (list, tuple)):
        current = current[0]
    if not isinstance(current, (list, tuple)) or len(current) == 0:
        raise RuntimeError("Unexpected attention structure in generation output.")
    if any(t is None for t in current):
        raise RuntimeError(
            "Attention tensors contain None. Try --attn_implementation eager or sdpa."
        )
    if not all(torch.is_tensor(t) for t in current):
        raise RuntimeError("Unexpected non-tensor entry in attention outputs.")
    return list(current)


def run_embedding_and_attention(
    encoder,
    model_inputs: Dict[str, torch.Tensor],
    save_dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.inference_mode():
        output = encoder.model.generate(
            **model_inputs,
            max_new_tokens=1,
            output_hidden_states=True,
            output_attentions=True,
            return_dict_in_generate=True,
            pad_token_id=encoder.processor.tokenizer.eos_token_id,
        )

    embedding = output.hidden_states[0][-1][:, -1, :].squeeze(0).detach().cpu().float()

    if not hasattr(output, "attentions") or output.attentions is None:
        raise RuntimeError(
            "Model did not return attentions. Try --attn_implementation eager."
        )

    per_layer = _flatten_first_generation_step_attentions(output.attentions)
    # Each tensor is expected as [B, H, S, S]. We save as [L, H, S, S].
    stacked = torch.stack(
        [layer_attn.squeeze(0).detach().to("cpu", dtype=save_dtype) for layer_attn in per_layer],
        dim=0,
    )
    return embedding, stacked


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def ids_to_string_with_image_tokens(model_inputs, encoder):
    """
    Convert model_inputs['input_ids'] to readable string, replacing visual spans with <image>.
    """
    ids = model_inputs["input_ids"][0]  # [S]
    tok = encoder.processor.tokenizer
    cfg = encoder.model.language_model.config

    vision_start_id = cfg.vision_start_token_id
    vision_end_id = cfg.vision_end_token_id
    image_token_id = cfg.image_token_id

    out = []
    i = 0
    n = ids.numel()

    while i < n:
        tid = int(ids[i].item())

        # collapse one full vision block to a single <image>
        if tid == vision_start_id:
            j = i + 1
            while j < n and int(ids[j].item()) != vision_end_id:
                j += 1

            if j < n:
                out.append("<image>")
                i = j + 1
                continue
            else:
                # malformed: no end token; fallback
                out.append("<image>")
                break

        # if standalone visual placeholder appears, map to <image>
        if tid == image_token_id:
            out.append("<image>")
            i += 1
            continue

        # regular token
        out.append(tok.decode([tid], skip_special_tokens=False))
        i += 1

    text = "".join(out)

    # optional cleanup for readable spacing
    text = text.replace("  ", " ")
    return text


def get_visual_span(input_ids, vision_start_id, vision_end_id):
    ids = input_ids[0]  # [S]
    s = (ids == vision_start_id).nonzero(as_tuple=True)[0].item()
    e = (ids == vision_end_id).nonzero(as_tuple=True)[0].item()
    # visual placeholders are in (s, e)
    return s + 1, e  # [start, end) python slice

def flat_to_thw(k, H_llm, W_llm):
    hw = H_llm * W_llm
    t = k // hw
    rem = k % hw
    r = rem // W_llm
    c = rem % W_llm
    return int(t), int(r), int(c)

def build_visual_index_map(model_inputs, encoder):
    # config + inputs
    grid = model_inputs["image_grid_thw"][0]   # (T, H, W)
    T, H, W = map(int, grid.tolist())
    sms = int(encoder.model.language_model.config.spatial_merge_size)

    T_llm = T
    H_llm = H // sms
    W_llm = W // sms
    N_vis = T_llm * H_llm * W_llm

    cfg = encoder.model.language_model.config
    vstart, vend = get_visual_span(
        model_inputs["input_ids"],
        cfg.vision_start_token_id,
        cfg.vision_end_token_id,
    )

    # absolute token positions in full sequence for visual block
    abs_positions = torch.arange(vstart, vstart + N_vis, device=model_inputs["input_ids"].device)

    # map each visual token to (t, h, w)
    thw = [flat_to_thw(k, H_llm, W_llm) for k in range(N_vis)]

    return {
        "visual_start": vstart,
        "visual_end_exclusive": vstart + N_vis,
        "T_llm": T_llm,
        "H_llm": H_llm,
        "W_llm": W_llm,
        "N_vis": N_vis,
        "abs_positions": abs_positions,  # token idx in full sequence
        "thw_map": thw,                  # list of (t,r,c), len N_vis
    }


def upsample_attn_lh_hw(
    attn_lh_hw: torch.Tensor,          # [L, Hh, h, w]
    out_hw: Tuple[int, int],           # (H', W')
    mode: str = "bilinear",
    align_corners: bool = False,
) -> torch.Tensor:
    if attn_lh_hw.ndim != 4:
        raise ValueError(f"Expected [L,H,h,w], got {tuple(attn_lh_hw.shape)}")
    L, Hh, h, w = attn_lh_hw.shape
    x = attn_lh_hw.reshape(L * Hh, 1, h, w)
    x = F.interpolate(x, size=out_hw, mode=mode, align_corners=align_corners)
    return x.reshape(L, Hh, out_hw[0], out_hw[1])


def unpatchify_qwen2vl_pixel_values(
    pixel_values: torch.Tensor,        # [t*h*w, C*tp*ps*ps]
    image_grid_thw: torch.Tensor,      # [3] = (t, h, w) from model_inputs["image_grid_thw"][0]
    C: int = 3,
    temporal_patch_size: int = 2,
    patch_size: int = 14,
    merge_size: int = 2,
) -> torch.Tensor:
    """
    Inverse of Qwen2VLImageProcessor flattening.

    Returns:
        frames_tchw: [T, C, H', W']
        where T = t * temporal_patch_size
              H' = h * patch_size
              W' = w * patch_size
    """
    if pixel_values.ndim != 2:
        raise ValueError(f"Expected pixel_values [N,D], got {tuple(pixel_values.shape)}")
    if image_grid_thw.numel() != 3:
        raise ValueError(f"Expected image_grid_thw [3], got {tuple(image_grid_thw.shape)}")

    t, h, w = map(int, image_grid_thw.tolist())
    tp, ps, m = temporal_patch_size, patch_size, merge_size

    if h % m != 0 or w % m != 0:
        raise ValueError(f"h,w must be divisible by merge_size. Got h={h}, w={w}, merge_size={m}")

    N_expected = t * h * w
    D_expected = C * tp * ps * ps
    if pixel_values.shape[0] != N_expected:
        raise ValueError(f"N mismatch: got {pixel_values.shape[0]}, expected {N_expected}")
    if pixel_values.shape[1] != D_expected:
        raise ValueError(f"D mismatch: got {pixel_values.shape[1]}, expected {D_expected}")

    h_m = h // m
    w_m = w // m

    # Inverse of:
    # patches.permute(0,1,4,7,5,8,3,2,6,9).reshape(batch, t*h*w, C*tp*ps*ps)
    x = pixel_values.view(t, h_m, w_m, m, m, C, tp, ps, ps)
    x = x.permute(0, 6, 5, 1, 3, 7, 2, 4, 8).contiguous()   # [t,tp,C,h_m,m,ps,w_m,m,ps]
    frames = x.view(t * tp, C, h_m * m * ps, w_m * m * ps)  # [T,C,H',W']
    return frames


def frames_tchw_to_pil_list(
    frames_tchw: torch.Tensor,         # [T, C, H, W], normalized
    mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073),  # OPENAI_CLIP_MEAN
    std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711),   # OPENAI_CLIP_STD
    clamp: bool = True,
) -> List[Image.Image]:
    if frames_tchw.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], got {tuple(frames_tchw.shape)}")
    if frames_tchw.shape[1] != 3:
        raise ValueError(f"Expected C=3, got C={frames_tchw.shape[1]}")

    x = frames_tchw.detach().float()
    device = x.device

    mean_t = torch.tensor(mean, device=device, dtype=x.dtype).view(1, 3, 1, 1)
    std_t = torch.tensor(std, device=device, dtype=x.dtype).view(1, 3, 1, 1)

    x = x * std_t + mean_t
    if clamp:
        x = x.clamp(0.0, 1.0)

    x_u8 = (x * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu()
    return [Image.fromarray(arr.numpy(), mode="RGB") for arr in x_u8]



def align_attn_to_frames(
    attn_lht_hw: torch.Tensor,    # [L, Hh, T, h, w] e.g. [28,28,4,9,15]
    frames_tchw: torch.Tensor,    # [T_img, C, H', W'] e.g. [8,3,252,420]
    temporal_patch_size: int = 2,
) -> torch.Tensor:
    """
    Returns:
        attn_per_frame: [T_img, H', W'] aligned to frames_tchw
    """
    if attn_lht_hw.ndim != 5:
        raise ValueError(f"Expected [L,H,T,h,w], got {tuple(attn_lht_hw.shape)}")
    if frames_tchw.ndim != 4:
        raise ValueError(f"Expected [T,C,H,W], got {tuple(frames_tchw.shape)}")

    L, Hh, T_attn, h, w = attn_lht_hw.shape
    T_img, _, Hp, Wp = frames_tchw.shape

    # 1) average across L and H
    attn_t_hw = attn_lht_hw.mean(dim=(0, 1))  # [T_attn, h, w]

    # 2) upsample each attention frame to image size
    # upsample_attn_lh_hw expects [L,H,h,w], so use L=T_attn, H=1
    attn_up = upsample_attn_lh_hw(
        attn_t_hw[:, None, :, :],  # [T_attn,1,h,w]
        out_hw=(Hp, Wp),
        mode="bilinear",
        align_corners=False,
    )[:, 0]  # [T_attn, Hp, Wp]

    # 3) temporal align: each attn token corresponds to temporal_patch_size frames
    attn_per_frame = attn_up.repeat_interleave(temporal_patch_size, dim=0)  # [T_attn*tp, Hp, Wp]

    if attn_per_frame.shape[0] != T_img:
        raise ValueError(
            f"Temporal mismatch: attn gives {attn_per_frame.shape[0]} frames, but frames_tchw has {T_img}."
        )

    return attn_per_frame