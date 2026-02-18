"""
Evaluates Charades TestOfTime temporal understanding benchmark with Qwen3-VL-Embedding.

For each sample:
1. Load N frames from event1_clip (first/earlier clip)
2. Load N frames from event2_clip (second/later clip)
3. Concatenate to get 2N frames total
4. Compute video embedding zv with Qwen3VLEmbedder
5. Compute text embeddings for caption (zt+) and distractor (zt-)
6. Check if sim(zv, zt+) > sim(zv, zt-)

Usage:
    python eval_qwen3vl_embedding.py --model_name_or_path /path/to/checkpoint [--debug]
"""
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import argparse
import json

import numpy as np
import pandas as pd
import PIL.Image
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shared.utils as su
from models.qwen3vl_embedding import Qwen3VLEmbedder
from utils.video import read_frames_decord


# Default paths
DEFAULT_MODEL_PATH = "/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B"
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "charades_testoftime.csv")
DEFAULT_CLIP_DIR = "/scratch/shared/beegfs/piyush/datasets/Charades/Charades_v1_480_cut_clips"


def load_clips_and_concatenate_as_pil_frames(
    clip1_path: str,
    clip2_path: str,
    n_frames_per_clip: int = 8,
):
    """
    Load frames from two clips, concatenate, and convert to a PIL image sequence.

    Args:
        clip1_path: Path to the first clip file.
        clip2_path: Path to the second clip file.
        n_frames_per_clip: Number of frames to sample from each clip.

    Returns:
        List[PIL.Image.Image] of length (2 * n_frames_per_clip), in temporal order.
    """
    frames_clip1 = read_frames_decord(clip1_path, num_frames=n_frames_per_clip)
    frames_clip2 = read_frames_decord(clip2_path, num_frames=n_frames_per_clip)
    frames = torch.cat([frames_clip1, frames_clip2], dim=0)  # (2N, C, H, W), uint8
    frames_hwc = frames.permute(0, 2, 3, 1)  # (2N, H, W, C)
    return [PIL.Image.fromarray(frame.cpu().numpy()) for frame in frames_hwc]


def convert_caption_to_sequence_of_events(caption: str) -> str:
    if " before " in caption:
        e1, e2 = caption.split(" before ")
        return f"1. {e1} \n2. {e2}"
    if " after " in caption:
        e2, e1 = caption.split(" after ")
        return f"1. {e1} \n2. {e2}"
    raise ValueError(f"Invalid caption: {caption}")


def evaluate_charades_testoftime_qwen3vlemb(
    model_name_or_path: str,
    csv_path: str,
    clip_dir: str,
    n_frames_per_clip: int = 8,
    debug: bool = False,
    device_map: str = "cuda:0",
    use_sequence_of_events: bool = False,
    torch_dtype: str = "float16",
    attn_implementation: str = "flash_attention_2",
):
    """
    Evaluate Qwen3-VL-Embedding on Charades TestOfTime.
    """
    if torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    elif torch_dtype == "float16":
        dtype = torch.float16
    elif torch_dtype == "float32":
        dtype = torch.float32
    else:
        raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")

    print(f"Loading model from: {model_name_or_path}")
    model = Qwen3VLEmbedder(
        model_name_or_path=model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        device_map=device_map,
        max_frames=n_frames_per_clip * 2,
    )
    su.misc.num_params(model.model)

    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    df["clip1_path"] = df["event1_clip"].apply(lambda x: os.path.join(clip_dir, x))
    df["clip2_path"] = df["event2_clip"].apply(lambda x: os.path.join(clip_dir, x))
    df["clip1_exists"] = df["clip1_path"].apply(os.path.exists)
    df["clip2_exists"] = df["clip2_path"].apply(os.path.exists)
    df["clips_exist"] = df["clip1_exists"] & df["clip2_exists"]

    missing_clips = int((~df["clips_exist"]).sum())
    if missing_clips > 0:
        print(f"Warning: {missing_clips} samples have missing clip files")

    df = df[df["clips_exist"]].reset_index(drop=True)
    print(f"Samples with valid clips: {len(df)}")

    if debug:
        np.random.seed(42)
        df = df.sample(min(50, len(df)), random_state=42).reset_index(drop=True)
        print(f"Debug mode: evaluating on {len(df)} samples")

    results = []
    correct = 0
    total = 0

    for i in su.log.tqdm_iterator(range(len(df)), desc="Evaluating Charades TestOfTime"):
        row = df.iloc[i]
        try:
            video_frames = load_clips_and_concatenate_as_pil_frames(
                clip1_path=row["clip1_path"],
                clip2_path=row["clip2_path"],
                n_frames_per_clip=n_frames_per_clip,
            )

            caption = (
                row["caption"]
                if not use_sequence_of_events
                else convert_caption_to_sequence_of_events(row["caption"])
            )
            distractor = (
                row["distractor_caption"]
                if not use_sequence_of_events
                else convert_caption_to_sequence_of_events(row["distractor_caption"])
            )

            zv = model.process([{"video": video_frames}], normalize=True).squeeze(0).cpu().float()
            zt_pos = model.process([{"text": caption}], normalize=True).squeeze(0).cpu().float()
            zt_neg = model.process([{"text": distractor}], normalize=True).squeeze(0).cpu().float()

            sim_pos = float(torch.dot(zv, zt_pos).item())
            sim_neg = float(torch.dot(zv, zt_neg).item())
            is_correct = sim_pos > sim_neg

            results.append(
                {
                    "video_id": row["video_id"],
                    "conjugate": row["conjugate"],
                    "event1_clip": row["event1_clip"],
                    "event2_clip": row["event2_clip"],
                    "caption": row["caption"],
                    "distractor_caption": row["distractor_caption"],
                    "sim_positive": sim_pos,
                    "sim_negative": sim_neg,
                    "correct": is_correct,
                }
            )

            if is_correct:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error processing {row['video_id']} ({row['event1_clip']}, {row['event2_clip']}): {e}")
            continue

    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'=' * 60}")
    print("Charades TestOfTime Evaluation Results")
    print(f"{'=' * 60}")
    print(f"Model: {model_name_or_path}")
    print(f"Total samples evaluated: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"{'=' * 60}")

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print("\nAccuracy by conjugate type:")
        for conjugate in results_df["conjugate"].unique():
            subset = results_df[results_df["conjugate"] == conjugate]
            conj_acc = subset["correct"].mean() * 100
            print(f"  {conjugate}: {conj_acc:.2f}% ({subset['correct'].sum()}/{len(subset)})")

    return {
        "accuracy": accuracy,
        "total": total,
        "correct": correct,
        "model_path": model_name_or_path,
        "results": results_df.to_dict("records") if len(results_df) > 0 else [],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Charades TestOfTime benchmark with Qwen3-VL-Embedding"
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to Qwen3-VL-Embedding checkpoint",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=DEFAULT_CSV_PATH,
        help="Path to the Charades TestOfTime CSV file",
    )
    parser.add_argument(
        "--clip_dir",
        type=str,
        default=DEFAULT_CLIP_DIR,
        help="Directory containing the clip files",
    )
    parser.add_argument(
        "--n_frames_per_clip",
        type=int,
        default=8,
        help="Frames sampled from each clip (total frames = 2 * this)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with subset of samples",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda:0",
        help="Device mapping for model loading",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Torch dtype for model loading",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation used by the model",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save results",
    )
    parser.add_argument(
        "--use_sequence_of_events",
        action="store_true",
        help="Convert captions to explicit ordered event lists",
    )
    args = parser.parse_args()

    results = evaluate_charades_testoftime_qwen3vlemb(
        model_name_or_path=args.model_name_or_path,
        csv_path=args.csv_path,
        clip_dir=args.clip_dir,
        n_frames_per_clip=args.n_frames_per_clip,
        debug=args.debug,
        device_map=args.device_map,
        use_sequence_of_events=args.use_sequence_of_events,
        torch_dtype=args.torch_dtype,
        attn_implementation=args.attn_implementation,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = os.path.basename(args.model_name_or_path.rstrip("/")) or "model"

    output_path = os.path.join(args.output_dir, f"charades_testoftime_{model_name}.json")
    summary = {
        "accuracy": results["accuracy"],
        "total": results["total"],
        "correct": results["correct"],
        "model_path": results["model_path"],
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    if results["results"]:
        results_df = pd.DataFrame(results["results"])
        csv_output_path = os.path.join(
            args.output_dir, f"charades_testoftime_{model_name}_detailed.csv"
        )
        results_df.to_csv(csv_output_path, index=False)
        print(f"Detailed results saved to: {csv_output_path}")


if __name__ == "__main__":
    main()
