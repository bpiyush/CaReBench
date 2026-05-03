"""Compute Qwen3-VL-Embedding video and text embeddings for the synthetic spatial-motion dataset."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Set, Tuple

import torch

import shared.utils as su


def find_dataset_root_with_index(user_root: str) -> Tuple[str, str]:
    """
    Return (artifact_root, index_json_path) where artifact_root is the directory
    that contains ``videos/`` and ``metadata/`` (i.e. parent of index.json).
    """
    user_root = os.path.abspath(os.path.expanduser(user_root))
    candidates = [
        os.path.join(user_root, "index.json"),
        os.path.join(user_root, "spatial_motion_dataset", "index.json"),
    ]
    for index_path in candidates:
        if os.path.isfile(index_path):
            artifact_root = os.path.dirname(index_path)
            return artifact_root, index_path
    raise FileNotFoundError(
        f"No index.json under {user_root!r} (tried {candidates}). "
        "Point --dataset_root at the folder that contains index.json "
        "or at its parent (with spatial_motion_dataset/)."
    )


def entry_signature(e: Dict[str, Any]) -> Tuple[str, str, str]:
    """Stable key for resume de-duplication."""
    return (e["modality"], e["video_id"], e["key"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Encode spatial-motion videos and all per-video text queries (Qwen3-VL-Embedding)."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B",
    )
    parser.add_argument("--model_name", type=str, default="qwen3vlemb")
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/SyntheticMotion",
        help="Directory that contains index.json or spatial_motion_dataset/index.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output .pt path (default: {model_path}/embs/{model_name}_spatial_motion_embeddings.pt)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Ignore an existing output file and recompute everything",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=16,
        help="Video frame cap passed to Qwen3VLEmbedder (per video).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=("flash_attention_2", "sdpa", "eager"),
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float16",
        choices=("float16", "bfloat16", "float32"),
        help="dtype for the embedding model weights.",
    )
    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    artifact_root, index_path = find_dataset_root_with_index(args.dataset_root)
    with open(index_path, "r") as f:
        index: Dict[str, Any] = json.load(f)

    videos_meta: List[Dict[str, Any]] = index.get("videos", [])
    print(f"Loaded index from {index_path} ({len(videos_meta)} videos).")

    save_dir = os.path.join(args.model_path, "embs")
    os.makedirs(save_dir, exist_ok=True)
    out_path = args.output_path or os.path.join(
        save_dir, f"{args.model_name}_spatial_motion_embeddings.pt"
    )

    done: Set[Tuple[str, str, str]] = set()
    entries: List[Dict[str, Any]] = []

    if os.path.isfile(out_path) and not args.no_resume:
        try:
            prev = torch.load(out_path, map_location="cpu", weights_only=False)
        except TypeError:
            prev = torch.load(out_path, map_location="cpu")
        entries = list(prev.get("entries", []))
        done = {entry_signature(e) for e in entries}
        print(f"Resume: loaded {len(entries)} entries from {out_path}")

    from models.qwen3vl_embedding import Qwen3VLEmbedder

    model = Qwen3VLEmbedder(
        model_name_or_path=args.model_path,
        device_map="cuda",
        torch_dtype=torch_dtype,
        attn_implementation=args.attn_implementation,
    )
    su.misc.num_params(model.model)

    n_videos = len(videos_meta)
    for i in su.log.tqdm_iterator(range(n_videos), desc="spatial_motion embeddings (qwen3vl)"):
        row = videos_meta[i]
        video_id = row["video_id"]
        rel = row["video_file"]
        video_path = os.path.join(artifact_root, rel)
        if not os.path.isfile(video_path):
            print(f"Missing video, skip {video_id}: {video_path}")
            continue

        queries = row.get("queries") or []

        vid_sig = ("video", video_id, video_path)
        if vid_sig not in done:
            try:
                z = model.process(
                    [{"video": video_path, "max_frames": args.max_frames}],
                    normalize=False,
                ).squeeze(0).cpu().float()
                z = torch.nn.functional.normalize(z, dim=-1)
                entries.append(
                    {
                        "modality": "video",
                        "key": video_path,
                        "video_id": video_id,
                        "embedding": z,
                        "video_file": rel,
                        "label": row.get("label"),
                        "sub_label": row.get("sub_label"),
                    }
                )
                done.add(vid_sig)
            except Exception as ex:
                print(f"Error encoding video {video_id}: {ex}")

        for q in queries:
            q_sig = ("text", video_id, q)
            if q_sig in done:
                continue
            try:
                zt = model.process([{"text": q}], normalize=False).squeeze(0).cpu().float()
                zt = torch.nn.functional.normalize(zt, dim=-1)
                entries.append(
                    {
                        "modality": "text",
                        "key": q,
                        "video_id": video_id,
                        "embedding": zt,
                        "video_file": rel,
                        "label": row.get("label"),
                        "sub_label": row.get("sub_label"),
                    }
                )
                done.add(q_sig)
            except Exception as ex:
                print(f"Error encoding text for {video_id!r}: {q!r}: {ex}")

    payload = {
        "dataset_root": artifact_root,
        "index_json": index_path,
        "model_path": args.model_path,
        "model_name": args.model_name,
        "max_frames": args.max_frames,
        "attn_implementation": args.attn_implementation,
        "torch_dtype": args.torch_dtype,
        "index_summary": {
            "total_videos": index.get("total_videos"),
            "frame_size": index.get("frame_size"),
            "fps": index.get("fps"),
        },
        "entries": entries,
    }
    torch.save(payload, out_path)
    print(f"Saved {len(entries)} entries to {out_path}")


if __name__ == "__main__":
    main()
