#!/usr/bin/env python
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fire
import numpy as np
import torch
import yaml
from datasets import load_dataset
from PIL import Image, ImageDraw

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models.tarsier2.dataset.tarsier_datamodule import init_processor
from models.tarsier2.dataset.utils import format_one_sample
from utils.model import EOL_PROMPTS


class Tarsier2ChiralPairDataset:
    def __init__(
        self,
        rows,
        super_processor,
        text_prompt_template: str,
        video_prompt_template: str,
    ):
        self.rows = rows
        self.super_processor = super_processor
        self.text_prompt_template = text_prompt_template
        self.video_prompt_template = video_prompt_template
        if "<sent>" not in self.text_prompt_template:
            raise ValueError("text_prompt_template must contain '<sent>'.")
        if "<video>" not in self.video_prompt_template:
            raise ValueError("video_prompt_template must contain '<video>'.")

    @staticmethod
    def _unbatch_tensors(sample_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in sample_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            out[key] = value[0].contiguous() if value.dim() > 0 and value.size(0) == 1 else value.contiguous()
        return out

    def _encode_video_only(self, video_path: str) -> Dict[str, torch.Tensor]:
        sample = format_one_sample(media_file=str(video_path), prompt=self.video_prompt_template)
        sample = self.super_processor(sample)
        return self._unbatch_tensors(sample)

    def _encode_text_only(self, caption: str) -> Dict[str, torch.Tensor]:
        prompt = self.text_prompt_template.replace("<sent>", str(caption))
        sample = format_one_sample(media_file=None, prompt=prompt)
        sample = self.super_processor(sample)
        return self._unbatch_tensors(sample)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        return {
            "video0": self._encode_video_only(row["video0"]),
            "text0": self._encode_text_only(row["sent0"]),
            "video1": self._encode_video_only(row["video_hard_neg"]),
            "text1": self._encode_text_only(row["sent_hard_neg"]),
        }


@dataclass
class DataCollatorForTarsier2Pairs:
    pad_token_id: int
    pad_to_multiple_of: Optional[int] = 8

    def _pad_1d(self, values: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(v.size(0) for v in values)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        padded = []
        for v in values:
            if v.size(0) == max_len:
                padded.append(v)
                continue
            pad = torch.full((max_len - v.size(0),), pad_value, dtype=v.dtype)
            padded.append(torch.cat([pad, v], dim=0))
        return torch.stack(padded, dim=0)

    @staticmethod
    def _merge_tensor_list(values: List[torch.Tensor]) -> torch.Tensor:
        try:
            return torch.stack(values, dim=0)
        except RuntimeError:
            return torch.cat(values, dim=0)

    def _collate_stream(self, stream_samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batched: Dict[str, torch.Tensor] = {}
        for key in stream_samples[0].keys():
            values = [x[key] for x in stream_samples]
            if key == "input_ids":
                batched[key] = self._pad_1d(values, self.pad_token_id)
            elif key == "attention_mask":
                batched[key] = self._pad_1d(values, 0)
            elif key == "labels":
                batched[key] = self._pad_1d(values, -100)
            else:
                batched[key] = self._merge_tensor_list(values)
        return batched

    def __call__(self, features: List[Dict[str, Dict[str, torch.Tensor]]]):
        out = {}
        for key in ("video0", "text0", "video1", "text1"):
            out[key] = self._collate_stream([f[key] for f in features])
        return out


def _read_video_frames_cv2(video_path: str, num_frames: int) -> List[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1
    sample_idxs = np.linspace(0, max(0, total_frames - 1), num=num_frames, dtype=np.int64)

    frames: List[np.ndarray] = []
    for frame_idx in sample_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from {video_path}")
    return frames


def _build_frame_grid(frames: List[np.ndarray], rows: int, cols: int, tile_size: Tuple[int, int]) -> Image.Image:
    tw, th = tile_size
    canvas = Image.new("RGB", (cols * tw, rows * th), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for i in range(min(len(frames), rows * cols)):
        r = i // cols
        c = i % cols
        frame = Image.fromarray(frames[i]).resize((tw, th))
        canvas.paste(frame, (c * tw, r * th))
        draw.text((c * tw + 6, r * th + 6), str(i), fill=(255, 255, 0))
    return canvas


def _save_gif(frames: List[np.ndarray], out_path: str, fps: int = 4):
    try:
        import imageio.v2 as imageio
    except ImportError:
        return False
    imageio.mimsave(out_path, frames, duration=max(1, 1000 // max(1, fps)))
    return True


@dataclass
class ShapeSummary:
    key: str
    shape: Tuple[int, ...]
    dtype: str


def _tensor_shapes(d: Dict) -> List[ShapeSummary]:
    out: List[ShapeSummary] = []
    for k, v in d.items():
        if hasattr(v, "shape"):
            out.append(ShapeSummary(key=k, shape=tuple(v.shape), dtype=str(v.dtype)))
    return out


def main(
    model_name_or_path: str = "/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115",
    data_path: str = "data/generated-chiral-pairs-v1.csv",
    output_dir: str = "outputs/tarsier2_vlemb_data_debug",
    num_rows: int = 4,
    batch_size: int = 2,
    frames_per_video: int = 8,
    architecture: str = "Tarsier2ForConditionalGeneration",
    base_config_name: str = "default_config.yaml",
):
    del architecture
    os.makedirs(output_dir, exist_ok=True)
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    data = load_dataset("csv", data_files=data_path)["train"]
    n = min(num_rows, len(data))
    rows = data.select(range(n))

    config_path = os.path.join(os.path.dirname(__file__), "..", "models", "tarsier2", base_config_name)
    with open(config_path, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)
    super_processor = init_processor(model_name_or_path, base_config)
    tokenizer = super_processor.processor.tokenizer

    dataset = Tarsier2ChiralPairDataset(
        rows=rows,
        super_processor=super_processor,
        text_prompt_template=EOL_PROMPTS["text"],
        video_prompt_template=EOL_PROMPTS["video"],
    )
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = DataCollatorForTarsier2Pairs(pad_token_id=pad_token_id)

    report = {"rows": [], "single_examples": [], "batch": {}}

    # 1) Single-item checks
    for i in range(n):
        item = dataset[i]
        v0_shapes = _tensor_shapes(item["video0"])
        t0_shapes = _tensor_shapes(item["text0"])
        v1_shapes = _tensor_shapes(item["video1"])
        t1_shapes = _tensor_shapes(item["text1"])
        report["single_examples"].append(
            {
                "index": i,
                "video0_shapes": [s.__dict__ for s in v0_shapes],
                "text0_shapes": [s.__dict__ for s in t0_shapes],
                "video1_shapes": [s.__dict__ for s in v1_shapes],
                "text1_shapes": [s.__dict__ for s in t1_shapes],
            }
        )
        print(f"[item {i}] video0 keys: {sorted(item['video0'].keys())}")
        print(f"[item {i}] text0 keys: {sorted(item['text0'].keys())}")
        print(f"[item {i}] video1 keys: {sorted(item['video1'].keys())}")
        print(f"[item {i}] text1 keys: {sorted(item['text1'].keys())}")

    # 2) Collated batch checks
    batch_items = [dataset[i] for i in range(min(batch_size, n))]
    batch = collator(batch_items)

    for stream_key in ("video0", "text0", "video1", "text1"):
        assert stream_key in batch, f"Missing {stream_key} in collated batch"
        assert "input_ids" in batch[stream_key], f"Missing input_ids in {stream_key}"
        assert "attention_mask" in batch[stream_key], f"Missing attention_mask in {stream_key}"
        assert batch[stream_key]["input_ids"].ndim == 2, "input_ids must be [B, T]"
        assert batch[stream_key]["attention_mask"].ndim == 2, "attention_mask must be [B, T]"
        assert batch[stream_key]["input_ids"].shape[0] == min(batch_size, n), "Batch dimension mismatch"

    report["batch"]["video0_shapes"] = [s.__dict__ for s in _tensor_shapes(batch["video0"])]
    report["batch"]["text0_shapes"] = [s.__dict__ for s in _tensor_shapes(batch["text0"])]
    report["batch"]["video1_shapes"] = [s.__dict__ for s in _tensor_shapes(batch["video1"])]
    report["batch"]["text1_shapes"] = [s.__dict__ for s in _tensor_shapes(batch["text1"])]
    print("[batch] video0 shapes:", report["batch"]["video0_shapes"])
    print("[batch] text0 shapes:", report["batch"]["text0_shapes"])
    print("[batch] video1 shapes:", report["batch"]["video1_shapes"])
    print("[batch] text1 shapes:", report["batch"]["text1_shapes"])

    # 3) Save visualizations from raw videos for manual verification
    for i in range(n):
        row = rows[i]
        row_info = {
            "index": i,
            "sent0": row["sent0"],
            "sent_hard_neg": row["sent_hard_neg"],
            "video0": row["video0"],
            "video_hard_neg": row["video_hard_neg"],
        }

        for label, video_path in (("video0", row["video0"]), ("video_hard_neg", row["video_hard_neg"])):
            try:
                frames = _read_video_frames_cv2(video_path, frames_per_video)
                grid = _build_frame_grid(frames, rows=2, cols=max(1, frames_per_video // 2), tile_size=(256, 144))
                png_path = os.path.join(vis_dir, f"row{i}_{label}_grid.png")
                grid.save(png_path)
                row_info[f"{label}_grid_png"] = png_path

                gif_path = os.path.join(vis_dir, f"row{i}_{label}.gif")
                gif_saved = _save_gif(frames, gif_path)
                row_info[f"{label}_gif"] = gif_path if gif_saved else None
            except Exception as exc:
                row_info[f"{label}_error"] = str(exc)

        report["rows"].append(row_info)

    report_path = os.path.join(output_dir, "pipeline_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nSaved report to: {report_path}")
    print(f"Saved visualizations to: {vis_dir}")
    print("Data pipeline shape checks completed successfully.")


if __name__ == "__main__":
    fire.Fire(main)
