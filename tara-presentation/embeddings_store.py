"""Embedding cache under /work + in-memory FAISS index."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import torch

from config import EMBEDDINGS_DIR
from datasets import VideoEntry


@dataclass
class SearchResult:
    video_id: str
    score: float
    video_path: str
    preview_path: str
    caption: str


def cache_path(model_id: str, dataset: str) -> Path:
    """One .pt per model×dataset: dict video_id -> embedding (+ meta)."""
    d = EMBEDDINGS_DIR / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{dataset}.pt"


def load_feature_cache(model_id: str, dataset: str) -> tuple[dict[str, torch.Tensor], dict[str, dict]]:
    path = cache_path(model_id, dataset)
    if not path.exists():
        return {}, {}
    data = torch.load(path, weights_only=False)
    if isinstance(data, dict) and "embeddings" in data:
        embs = {k: v.float().cpu() for k, v in data["embeddings"].items()}
        meta = data.get("meta", {})
        return embs, meta
    # legacy: bare dict of tensors
    return {k: v.float().cpu() for k, v in data.items()}, {}


def save_feature_cache(
    model_id: str,
    dataset: str,
    embeddings: dict[str, torch.Tensor],
    meta: dict[str, dict],
) -> Path:
    path = cache_path(model_id, dataset)
    payload = {
        "embeddings": {k: v.detach().float().cpu() for k, v in embeddings.items()},
        "meta": meta,
    }
    torch.save(payload, path)
    return path


class EmbeddingIndex:
    def __init__(self):
        self.model_id: str | None = None
        self.dataset: str | None = None
        self.split: str | None = None
        self.sample_pct: float | None = None
        self.embeddings: np.ndarray | None = None
        self.entries: list[VideoEntry] = []
        self.index: faiss.IndexFlatIP | None = None

    @property
    def ready(self) -> bool:
        return self.embeddings is not None and self.index is not None and len(self.entries) > 0

    def build_from_entries(
        self,
        model_id: str,
        dataset: str,
        split: str,
        sample_pct: float,
        entries: list[VideoEntry],
        emb_dict: dict[str, torch.Tensor],
    ) -> None:
        vectors = []
        kept: list[VideoEntry] = []
        for e in entries:
            if e.video_id not in emb_dict:
                continue
            vectors.append(emb_dict[e.video_id])
            kept.append(e)
        if not vectors:
            raise RuntimeError("No embeddings available for selected videos")

        emb_np = torch.stack(vectors, dim=0).cpu().numpy().astype(np.float32)
        faiss.normalize_L2(emb_np)
        index = faiss.IndexFlatIP(emb_np.shape[1])
        index.add(emb_np)

        self.model_id = model_id
        self.dataset = dataset
        self.split = split
        self.sample_pct = sample_pct
        self.embeddings = emb_np
        self.entries = kept
        self.index = index

    def search(self, query: torch.Tensor, top_k: int = 12) -> list[SearchResult]:
        if not self.ready or self.index is None:
            return []
        q = query.cpu().numpy().astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, len(self.entries)))
        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            entry = self.entries[idx]
            results.append(
                SearchResult(
                    video_id=entry.video_id,
                    score=float(score),
                    video_path=entry.video_path,
                    preview_path=entry.preview_path,
                    caption=entry.caption,
                )
            )
        return results

    def entry_by_id(self, video_id: str) -> VideoEntry | None:
        for e in self.entries:
            if e.video_id == video_id:
                return e
        return None
