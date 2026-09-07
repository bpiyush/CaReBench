"""Load MSRVTT and CiA video subsets for retrieval."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import (
    CIA_ROOT,
    CIA_SOURCES,
    MSRVTT_ANNOTATIONS,
    MSRVTT_TRAIN_LIST,
    MSRVTT_VAL_LIST,
    MSRVTT_VIDEO_DIR,
    VIDEO_DIRS,
)


@dataclass
class VideoEntry:
    video_id: str
    video_path: str
    preview_path: str
    caption: str = ""


def _read_id_list(path: Path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _load_msrvtt_captions() -> dict[str, list[str]]:
    with open(MSRVTT_ANNOTATIONS) as f:
        data = json.load(f)
    captions: dict[str, list[str]] = {}
    for ann in data["annotations"]:
        captions.setdefault(ann["image_id"], []).append(ann["caption"])
    return captions


def load_msrvtt_split(split: str) -> list[VideoEntry]:
    """Load MSRVTT train or val (test) split."""
    split = split.lower()
    if split == "train":
        video_ids = _read_id_list(MSRVTT_TRAIN_LIST)
    elif split in ("val", "validation", "test"):
        video_ids = _read_id_list(MSRVTT_VAL_LIST)
    else:
        raise ValueError(f"Unknown MSRVTT split: {split}")

    captions = _load_msrvtt_captions()
    entries: list[VideoEntry] = []
    for vid in video_ids:
        full_path = MSRVTT_VIDEO_DIR / f"{vid}.mp4"
        if not full_path.exists():
            continue
        caps = captions.get(vid, [])
        entries.append(
            VideoEntry(
                video_id=vid,
                video_path=str(full_path),
                # Display uses downsized originals (not msrvtt_2fps_224 — that is 2 FPS).
                preview_path=str(full_path),
                caption=caps[0] if caps else "",
            )
        )
    return entries


def _resolve_cia_video_path(row: pd.Series, source: str) -> str | None:
    if source == "ssv2":
        path = VIDEO_DIRS["ssv2"] / f"{row['id']}.webm"
    elif source == "epic":
        participant = row["participant_id"]
        clip_name = row["path_id"].split("/")[-1]
        path = VIDEO_DIRS["epic"] / participant / "videos" / f"{clip_name}.MP4"
    elif source == "charades":
        path = VIDEO_DIRS["charades"] / f"{row['item_id']}.mp4"
    else:
        return None
    return str(path) if path.exists() else None


def _cia_text(row: pd.Series, source: str) -> str:
    if source == "ssv2":
        return str(row.get("template", row.get("label", "")))
    if source == "epic":
        return str(row.get("narration", row.get("verb", "")))
    return str(row.get("label", row.get("verb", "")))


def load_cia_split(split_name: str) -> list[VideoEntry]:
    """Load a CiA split, deduplicated by video path."""
    if split_name not in CIA_SOURCES:
        raise ValueError(f"Unknown CiA split: {split_name}")

    source, split = CIA_SOURCES[split_name]
    csv_path = CIA_ROOT / f"cia-{source}-{split}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CiA CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    seen_paths: set[str] = set()
    entries: list[VideoEntry] = []

    for _, row in df.iterrows():
        video_path = _resolve_cia_video_path(row, source)
        if video_path is None or video_path in seen_paths:
            continue
        seen_paths.add(video_path)

        if source == "ssv2":
            video_id = str(row["id"])
        elif source == "epic":
            video_id = str(row["narration_id"])
        else:
            video_id = str(row["item_id"])

        entries.append(
            VideoEntry(
                video_id=video_id,
                video_path=video_path,
                preview_path=video_path,
                caption=_cia_text(row, source),
            )
        )
    return entries


def load_videos(dataset: str, split: str) -> list[VideoEntry]:
    dataset = dataset.lower()
    if dataset == "msrvtt":
        return load_msrvtt_split(split)
    if dataset == "cia":
        return load_cia_split(split)
    raise ValueError(f"Unknown dataset: {dataset}")


def load_all_videos(dataset: str) -> list[VideoEntry]:
    """Load every split for a dataset, deduped by video_id (first wins)."""
    from config import DATASETS

    dataset = dataset.lower()
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset}")
    seen: set[str] = set()
    out: list[VideoEntry] = []
    for split in DATASETS[dataset]["splits"]:
        for e in load_videos(dataset, split):
            if e.video_id in seen:
                continue
            seen.add(e.video_id)
            out.append(e)
    return out
