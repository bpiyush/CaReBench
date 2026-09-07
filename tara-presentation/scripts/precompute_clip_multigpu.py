#!/usr/bin/env python3
"""Multi-GPU CLIP (avgpool) feature precompute for the TARA demo cache.

Shards videos across free (or all) GPUs, writes per-GPU shards, then merges
into embeddings/clip/{dataset}.pt.

Usage:
  python scripts/precompute_clip_multigpu.py --datasets cia msrvtt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    CAREBENCH_PYTHON,
    CLIP_PATH,
    EMBEDDINGS_DIR,
    MODELS,
)
from datasets import load_all_videos  # noqa: E402
from embeddings_store import load_feature_cache, save_feature_cache  # noqa: E402


def free_gpu_ids(mem_free_mib: int = 1500) -> list[int]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return [0] if torch.cuda.is_available() else []
    free: list[int] = []
    for line in out.strip().splitlines():
        idx_s, used_s = [x.strip() for x in line.split(",")]
        if int(float(used_s)) < mem_free_mib:
            free.append(int(idx_s))
    return free


def shard_path(dataset: str, shard_id: int) -> Path:
    d = EMBEDDINGS_DIR / "clip" / "_shards" / dataset
    d.mkdir(parents=True, exist_ok=True)
    return d / f"shard_{shard_id:02d}.pt"


def run_worker(
    *,
    dataset: str,
    shard_id: int,
    gpu: int,
    video_ids: list[str],
    paths: list[str],
    captions: list[str],
) -> tuple[Path, subprocess.Popen, Path, object]:
    """Spawn a single-GPU worker over a list of videos."""
    out = shard_path(dataset, shard_id)
    # Persist job list next to shard for the worker.
    job = out.with_suffix(".job.pt")
    torch.save(
        {
            "video_ids": video_ids,
            "paths": paths,
            "captions": captions,
            "out": str(out),
            "model_path": str(CLIP_PATH),
        },
        job,
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = f"{ROOT}:{env.get('PYTHONPATH', '')}"
    env["TOKENIZERS_PARALLELISM"] = "false"
    cmd = [
        str(CAREBENCH_PYTHON),
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--job",
        str(job),
    ]
    log = out.with_suffix(".log")
    print(f"[launcher] GPU {gpu} shard {shard_id}: {len(video_ids)} videos → {out}")
    # Line-buffered log; leave open so the child fd stays valid.
    log_f = open(log, "w", encoding="utf-8", buffering=1)
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        cwd=str(ROOT),
    )
    return out, proc, log, log_f


def worker_main(job_path: Path) -> None:
    from embedders import ClipAvgPoolEmbedder

    print(f"[worker] start job={job_path}", flush=True)
    job = torch.load(job_path, weights_only=False)
    out = Path(job["out"])
    emb: dict[str, torch.Tensor] = {}
    meta: dict[str, dict] = {}
    print(f"[worker] loading CLIP on cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '?')}", flush=True)
    embedder = ClipAvgPoolEmbedder(job["model_path"])
    embedder.load()
    print("[worker] model ready", flush=True)
    n = len(job["video_ids"])
    t0 = time.time()
    for i, (vid, path, cap) in enumerate(
        zip(job["video_ids"], job["paths"], job["captions"])
    ):
        try:
            z = embedder.encode_video(path)
            emb[vid] = z
            meta[vid] = {
                "video_path": path,
                "preview_path": path,
                "caption": cap or "",
            }
        except Exception as exc:  # noqa: BLE001
            print(f"[worker] skip {vid}: {exc}", flush=True)
        if (i + 1) % 10 == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 1e-6)
            print(
                f"[worker] {i + 1}/{n} · {rate:.2f} vid/s · cached={len(emb)}",
                flush=True,
            )
            torch.save({"embeddings": emb, "meta": meta}, out)
    embedder.close()
    torch.save({"embeddings": emb, "meta": meta}, out)
    print(f"[worker] done → {out} ({len(emb)} embeddings)", flush=True)


def merge_shards(dataset: str, shard_outs: list[Path]) -> Path:
    emb, meta = load_feature_cache("clip", dataset)
    before = len(emb)
    for sp in shard_outs:
        if not sp.exists():
            print(f"[merge] missing shard {sp}")
            continue
        data = torch.load(sp, weights_only=False)
        for k, v in data.get("embeddings", {}).items():
            emb[k] = v.float().cpu()
        for k, v in data.get("meta", {}).items():
            meta[k] = v
    path = save_feature_cache("clip", dataset, emb, meta)
    print(f"[merge] {dataset}: {before} → {len(emb)} embeddings → {path}")
    return path


def precompute_dataset(
    dataset: str,
    gpus: list[int],
    *,
    split: str | None = None,
    shard_tag: str | None = None,
) -> None:
    from datasets import load_videos

    if split:
        entries = load_videos(dataset, split)
        tag = shard_tag or f"{dataset}_{split.replace('/', '-')}"
    else:
        entries = load_all_videos(dataset)
        tag = shard_tag or dataset

    emb, _ = load_feature_cache("clip", dataset)
    todo = [e for e in entries if e.video_id not in emb]
    print(
        f"[launcher] {dataset}"
        + (f"/{split}" if split else "")
        + f": {len(entries)} videos, {len(emb)} cached total, "
        f"{len(todo)} to encode on GPUs {gpus}"
    )
    if not todo:
        print(f"[launcher] {dataset}: nothing to do")
        return
    if not gpus:
        raise RuntimeError("No free GPUs available for CLIP precompute")

    # Even shard across GPUs
    shards: list[list] = [[] for _ in gpus]
    for i, e in enumerate(todo):
        shards[i % len(gpus)].append(e)

    procs = []
    outs: list[Path] = []
    log_handles = []
    for shard_id, (gpu, chunk) in enumerate(zip(gpus, shards)):
        if not chunk:
            continue
        out, proc, log, log_f = run_worker(
            dataset=tag,
            shard_id=shard_id,
            gpu=gpu,
            video_ids=[e.video_id for e in chunk],
            paths=[e.video_path for e in chunk],
            captions=[e.caption for e in chunk],
        )
        outs.append(out)
        procs.append((proc, log, gpu, shard_id))
        log_handles.append(log_f)
        # Stagger CLIP checkpoint loads across GPUs.
        time.sleep(1)

    failed = False
    for proc, log, gpu, shard_id in procs:
        rc = proc.wait()
        if rc != 0:
            failed = True
            print(f"[launcher] GPU {gpu} shard {shard_id} FAILED rc={rc}. Tail:")
            try:
                print(log.read_text(errors="replace")[-2000:])
            except Exception:  # noqa: BLE001
                pass
    for h in log_handles:
        try:
            h.close()
        except Exception:  # noqa: BLE001
            pass
    # Always merge into the dataset-level cache (cia.pt / msrvtt.pt).
    merge_shards(dataset, outs)
    if failed:
        raise RuntimeError(f"One or more shards failed for {dataset}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["cia", "msrvtt"])
    p.add_argument(
        "--split",
        type=str,
        default="",
        help="Optional single split (e.g. SSv2-Validation). Only used with one --datasets value.",
    )
    p.add_argument("--gpus", type=str, default="", help="Comma list, else auto free GPUs")
    p.add_argument("--mem-free-mib", type=int, default=1500)
    p.add_argument("--all-gpus", action="store_true", help="Use every GPU (ignore free check)")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--job", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()

    if args.worker:
        assert args.job is not None
        worker_main(args.job)
        return

    if "clip" not in MODELS:
        raise SystemExit("clip model missing from config.MODELS")

    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    elif args.all_gpus:
        n = torch.cuda.device_count()
        gpus = list(range(n))
    else:
        gpus = free_gpu_ids(args.mem_free_mib)
    print(f"[launcher] using GPUs: {gpus}")

    split = args.split.strip() or None
    if split and len(args.datasets) != 1:
        raise SystemExit("--split requires exactly one --datasets value")

    for ds in args.datasets:
        precompute_dataset(ds, gpus, split=split)


if __name__ == "__main__":
    main()
