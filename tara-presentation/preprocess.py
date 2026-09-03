"""Preprocessing pipeline with sample_pct and persistent /work cache."""
from __future__ import annotations

import random
import time
from typing import Callable

from pathlib import Path

from datasets import VideoEntry, load_videos
from embedders import EmbedderBackend, create_embedder
from embeddings_store import (
    EmbeddingIndex,
    load_feature_cache,
    save_feature_cache,
)
from preview import get_or_create_preview


ProgressCb = Callable[[float, str, float | None], None]
# progress fraction, message, eta_seconds


def _sample_entries(entries: list[VideoEntry], sample_pct: float, seed: int = 0) -> list[VideoEntry]:
    pct = max(0.1, min(100.0, float(sample_pct)))
    if pct >= 99.999:
        return list(entries)
    n = max(1, int(round(len(entries) * pct / 100.0)))
    rng = random.Random(seed)
    return rng.sample(list(entries), n)


def preprocess_videos(
    model_id: str,
    model_path: str,
    dataset: str,
    split: str,
    sample_pct: float = 100.0,
    sample_seed: int = 42,
    progress_cb: ProgressCb | None = None,
    index: EmbeddingIndex | None = None,
) -> EmbeddingIndex:
    """Embed videos (skipping cache hits), update /work .pt dict, build FAISS index."""
    store = index or EmbeddingIndex()
    all_entries = load_videos(dataset, split)
    if not all_entries:
        raise RuntimeError(f"No videos found for {dataset}/{split}")

    entries = _sample_entries(all_entries, sample_pct, seed=int(sample_seed))
    emb_dict, meta = load_feature_cache(model_id, dataset)

    # Resolve preview paths up front for cache hits (never use 2fps extracts).
    preview_key = f"{model_id}__{dataset}"
    for e in entries:
        m = meta.get(e.video_id, {})
        cached_preview = m.get("preview_path") or ""
        if cached_preview and "msrvtt_2fps_224" not in cached_preview and Path(cached_preview).exists():
            e.preview_path = cached_preview
        else:
            e.preview_path = e.video_path

    todo = [e for e in entries if e.video_id not in emb_dict]
    n_cached = len(entries) - len(todo)
    total = len(todo)

    if progress_cb:
        progress_cb(
            0.0 if total else 1.0,
            f"{n_cached} cached · {total} to encode (of {len(entries)} selected)",
            None,
        )

    if total > 0:
        embedder: EmbedderBackend = create_embedder(model_id, model_path)
        embedder.load()
        t0 = time.time()
        done = 0
        last_error: str | None = None
        for entry in todo:
            try:
                emb = embedder.encode_video(entry.video_path)
                preview = get_or_create_preview(
                    entry.video_path,
                    preview_key,
                    entry.video_id,
                    existing_preview=entry.preview_path,
                )
                entry.preview_path = preview
                emb_dict[entry.video_id] = emb
                meta[entry.video_id] = {
                    "video_path": entry.video_path,
                    "preview_path": preview,
                    "caption": entry.caption,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if progress_cb:
                    progress_cb(
                        done / total,
                        f"Skipped {entry.video_id}: {exc}",
                        None,
                    )
                done += 1
                continue

            done += 1
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remaining = (total - done) / max(rate, 1e-6)
            if progress_cb:
                progress_cb(
                    done / total,
                    f"Encoded {done}/{total} · {entry.video_id}",
                    remaining,
                )

            if done % 25 == 0 or done == total:
                save_feature_cache(model_id, dataset, emb_dict, meta)

        embedder.close()
        if not any(e.video_id in emb_dict for e in entries) and last_error:
            raise RuntimeError(
                f"All {total} video encodes failed. Last error: {last_error}"
            )
        save_feature_cache(model_id, dataset, emb_dict, meta)

    # Ensure smooth display previews (even for cache hits) and refresh meta
    final_entries: list[VideoEntry] = []
    for e in entries:
        if e.video_id not in emb_dict:
            continue
        m = meta.get(e.video_id, {})
        video_path = m.get("video_path", e.video_path)
        caption = m.get("caption", e.caption)
        preview = m.get("preview_path", e.preview_path)
        if (not preview) or ("msrvtt_2fps_224" in str(preview)) or (not Path(preview).exists()):
            preview = get_or_create_preview(video_path, preview_key, e.video_id)
            meta[e.video_id] = {
                "video_path": video_path,
                "preview_path": preview,
                "caption": caption,
            }
        final_entries.append(
            VideoEntry(
                video_id=e.video_id,
                video_path=video_path,
                preview_path=preview,
                caption=caption,
            )
        )
    save_feature_cache(model_id, dataset, emb_dict, meta)

    store.build_from_entries(
        model_id=model_id,
        dataset=dataset,
        split=split,
        sample_pct=sample_pct,
        entries=final_entries,
        emb_dict=emb_dict,
    )

    if progress_cb:
        progress_cb(1.0, f"Ready — {len(store.entries)} videos indexed", 0.0)
    return store
