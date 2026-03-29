"""Merge LoRA adapter weights into the base Qwen3-VL-Embedding model and save a full checkpoint."""
import os
import sys

import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from peft import PeftModel

from models.qwen3vl_embedding import Qwen3VLForEmbedding
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

# Default output: <fine_tuned_model>/merged_checkpoint/
MERGED_CHECKPOINT_SUBDIR = "merged_checkpoint"


def _has_adapter_config(path: str) -> bool:
    return os.path.isfile(os.path.join(path, "adapter_config.json"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load base Qwen3VL-Embedding + LoRA adapter, merge, and save to a new directory."
    )
    parser.add_argument(
        "-b",
        "--base_model",
        type=str,
        required=True,
        help="Path to base Qwen3-VL-Embedding checkpoint (full weights).",
    )
    parser.add_argument(
        "-f",
        "--fine_tuned_model",
        type=str,
        required=True,
        help="Path to LoRA output directory (contains adapter_config.json and adapter weights).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help=(
            f"Output directory for the merged model. "
            f"If omitted, writes to <fine_tuned_model>/{MERGED_CHECKPOINT_SUBDIR}/ "
            f"(i.e. inside the LoRA run directory)."
        ),
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Load weights in float16 (default: bfloat16).",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='device_map for from_pretrained (e.g. "auto", "cpu", "cuda:0").',
    )
    args = parser.parse_args()

    fine_tuned_root = os.path.abspath(os.path.expanduser(args.fine_tuned_model))

    if not _has_adapter_config(fine_tuned_root):
        raise FileNotFoundError(
            f"No adapter_config.json under {fine_tuned_root}. "
            "Pass the LoRA adapter directory from finetuning_qwen3vlemb_lora.py, not a merged full model."
        )

    if args.save_dir:
        save_dir = os.path.abspath(os.path.expanduser(args.save_dir))
    else:
        save_dir = os.path.join(fine_tuned_root, MERGED_CHECKPOINT_SUBDIR)
    os.makedirs(save_dir, exist_ok=True)

    torch_dtype = torch.float16 if args.fp16 else torch.bfloat16

    load_kw = dict(
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )
    if args.device_map == "cpu":
        load_kw["device_map"] = None
        load_kw["low_cpu_mem_usage"] = True
    else:
        load_kw["device_map"] = args.device_map

    print(f"Loading base model from {args.base_model} (dtype={torch_dtype}, device_map={args.device_map})...")
    base_model = Qwen3VLForEmbedding.from_pretrained(args.base_model, **load_kw)
    if args.device_map == "cpu":
        base_model = base_model.to("cpu")

    print(f"Loading LoRA adapter from {fine_tuned_root}...")
    model = PeftModel.from_pretrained(
        base_model,
        fine_tuned_root,
    )

    print("Merging LoRA into base weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {save_dir}")
    merged_model.save_pretrained(save_dir)

    # Prefer processor/tokenizer saved alongside the adapter run; fall back to base.
    try:
        processor = Qwen3VLProcessor.from_pretrained(fine_tuned_root)
        print(f"Saved processor/tokenizer from {fine_tuned_root}")
    except Exception as e:
        print(f"Could not load processor from adapter dir ({e}); using base_model.")
        processor = Qwen3VLProcessor.from_pretrained(args.base_model)

    processor.save_pretrained(save_dir)
    processor.tokenizer.save_pretrained(save_dir)

    print(f"Done. Merged checkpoint written to {save_dir}")
