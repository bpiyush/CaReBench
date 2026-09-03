"""FastAPI backend + static frontend for text-to-video search."""
from __future__ import annotations

import argparse
import re
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import (
    CACHE_ROOT,
    DATASETS,
    EXAMPLE_QUERIES,
    MODELS,
    STATIC_DIR,
    TOP_K_DEFAULT,
    DEFAULT_SAMPLE_PCT,
    DEFAULT_SAMPLE_SEED,
)
from embedders import create_embedder, release_gpu_memory
from embeddings_store import EmbeddingIndex, load_feature_cache
from gif_grid import build_gif
from preprocess import preprocess_videos

app = FastAPI(title="TARA Video Search")

_state: dict[str, Any] = {
    "index": EmbeddingIndex(),
    "embedder": None,
    "model_id": None,
    "model_path": None,
    "model_label": None,
    "dataset_id": None,
    "dataset_label": None,
    "split": None,
    "sample_pct": None,
    "sample_seed": None,
    "job_id": 0,
    "job": {
        "job_id": 0,
        "status": "idle",  # idle | running | done | error
        "progress": 0.0,
        "message": "",
        "eta_sec": None,
        "error": None,
        "started_at": None,
    },
}
_job_lock = threading.Lock()


class PreprocessRequest(BaseModel):
    model_id: str
    dataset_id: str
    split: str
    sample_pct: float = Field(default=DEFAULT_SAMPLE_PCT, ge=0.1, le=100.0)
    sample_seed: int = Field(default=DEFAULT_SAMPLE_SEED, ge=0)


class SearchRequest(BaseModel):
    query: str
    top_k: int = Field(default=TOP_K_DEFAULT, ge=1, le=48)


class RecordRequest(BaseModel):
    query: str = ""
    video_ids: list[str]
    scores: list[float] = []
    captions: list[str] = []
    duration: float = Field(default=4.0, ge=2.0, le=12.0)


def _set_job(**kwargs):
    with _job_lock:
        _state["job"].update(kwargs)


def _release_embedder() -> None:
    """Unload the in-memory search model and free GPU memory."""
    embedder = _state.get("embedder")
    if embedder is not None:
        try:
            embedder.close()
        except Exception:  # noqa: BLE001
            pass
    _state["embedder"] = None
    _state["model_id"] = None
    _state["model_path"] = None
    release_gpu_memory()


def _get_embedder(model_id: str, model_path: str):
    if (
        _state["embedder"] is not None
        and _state["model_id"] == model_id
        and _state["model_path"] == model_path
    ):
        return _state["embedder"]
    _release_embedder()
    emb = create_embedder(model_id, model_path)
    emb.load()
    _state["embedder"] = emb
    _state["model_id"] = model_id
    _state["model_path"] = model_path
    return emb


def _run_preprocess(req: PreprocessRequest, job_id: int):
    model = MODELS.get(req.model_id)
    dataset = DATASETS.get(req.dataset_id)
    if model is None or dataset is None:
        _set_job(job_id=job_id, status="error", error="Unknown model or dataset", progress=0.0)
        return
    if req.split not in dataset["splits"]:
        _set_job(job_id=job_id, status="error", error=f"Unknown split: {req.split}", progress=0.0)
        return

    def progress_cb(p: float, msg: str, eta: float | None):
        _set_job(job_id=job_id, progress=float(p), message=msg, eta_sec=eta)

    try:
        # Free GPU before loading the new preprocessing model.
        _release_embedder()
        _state["index"] = EmbeddingIndex()

        _set_job(
            job_id=job_id,
            status="running",
            progress=0.0,
            message="Loading model…",
            eta_sec=None,
            error=None,
            started_at=time.time(),
        )
        index = preprocess_videos(
            model_id=model["id"],
            model_path=model["path"],
            dataset=dataset["id"],
            split=req.split,
            sample_pct=req.sample_pct,
            sample_seed=req.sample_seed,
            progress_cb=progress_cb,
            index=_state["index"],
        )
        if _state["job_id"] != job_id:
            return  # superseded by a newer job

        _state["index"] = index
        _get_embedder(model["id"], model["path"])
        _state["model_label"] = model["label"]
        _state["dataset_id"] = dataset["id"]
        _state["dataset_label"] = dataset["label"]
        _state["split"] = req.split
        _state["sample_pct"] = req.sample_pct
        _state["sample_seed"] = req.sample_seed
        _set_job(
            job_id=job_id,
            status="done",
            progress=1.0,
            message=f"Ready — {len(index.entries)} videos",
            eta_sec=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        if _state["job_id"] == job_id:
            _set_job(job_id=job_id, status="error", error=str(exc), message=str(exc), progress=0.0)


@app.get("/api/config")
def api_config():
    return {
        "models": [{"id": m["id"], "label": m["label"]} for m in MODELS.values()],
        "datasets": [
            {"id": d["id"], "label": d["label"], "splits": d["splits"]}
            for d in DATASETS.values()
        ],
        "default_sample_pct": DEFAULT_SAMPLE_PCT,
        "default_sample_seed": DEFAULT_SAMPLE_SEED,
        "default_top_k": TOP_K_DEFAULT,
        "example_queries": EXAMPLE_QUERIES,
    }


@app.get("/api/session")
def api_session():
    idx: EmbeddingIndex = _state["index"]
    return {
        "ready": idx.ready,
        "model_id": _state["model_id"],
        "model_label": _state["model_label"],
        "dataset_id": _state["dataset_id"],
        "dataset_label": _state["dataset_label"],
        "split": _state["split"],
        "sample_pct": _state["sample_pct"],
        "sample_seed": _state["sample_seed"],
        "n_videos": len(idx.entries) if idx.ready else 0,
        "job_id": _state["job_id"],
        "job": dict(_state["job"]),
    }


@app.post("/api/reset")
def api_reset():
    """Release GPU model and clear session when user changes setup."""
    if _state["job"]["status"] == "running":
        raise HTTPException(409, "Cannot reset while preprocessing is running")
    _release_embedder()
    _state["index"] = EmbeddingIndex()
    _state["model_label"] = None
    _state["dataset_id"] = None
    _state["dataset_label"] = None
    _state["split"] = None
    _state["sample_pct"] = None
    _state["sample_seed"] = None
    _set_job(
        job_id=_state["job_id"],
        status="idle",
        progress=0.0,
        message="",
        eta_sec=None,
        error=None,
        started_at=None,
    )
    return {"ok": True}


@app.post("/api/preprocess")
def api_preprocess(req: PreprocessRequest):
    if _state["job"]["status"] == "running":
        raise HTTPException(409, "Preprocessing already running")

    _state["job_id"] += 1
    job_id = _state["job_id"]
    # Mark running synchronously so the client never sees a stale "done".
    _set_job(
        job_id=job_id,
        status="running",
        progress=0.0,
        message="Starting…",
        eta_sec=None,
        error=None,
        started_at=time.time(),
    )

    t = threading.Thread(target=_run_preprocess, args=(req, job_id), daemon=True)
    t.start()
    return {"ok": True, "job_id": job_id}


@app.get("/api/preprocess/status")
def api_preprocess_status():
    return dict(_state["job"])


@app.post("/api/search")
def api_search(req: SearchRequest):
    idx: EmbeddingIndex = _state["index"]
    if not idx.ready:
        raise HTTPException(400, "Run preprocessing first")
    query = (req.query or "").strip()
    if not query:
        raise HTTPException(400, "Empty query")
    embedder = _state["embedder"]
    if embedder is None:
        raise HTTPException(400, "Model not loaded")
    try:
        q = embedder.encode_text(query)
        results = idx.search(q, top_k=req.top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc
    return {
        "query": query,
        "results": [
            {
                "video_id": r.video_id,
                "score": round(r.score, 4),
                "caption": r.caption,
                "video_url": f"/api/video/{r.video_id}",
            }
            for r in results
        ],
    }


@app.get("/api/video/{video_id}")
def api_video(video_id: str):
    idx: EmbeddingIndex = _state["index"]
    entry = idx.entry_by_id(video_id)
    if entry is None:
        raise HTTPException(404, "Unknown video")
    preview = Path(entry.preview_path)
    original = Path(entry.video_path)
    if preview.exists() and "msrvtt_2fps_224" not in str(preview):
        path = preview
    elif original.exists():
        path = original
    else:
        raise HTTPException(404, f"File missing: {entry.video_path}")
    media = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
    return FileResponse(path, media_type=media, filename=path.name)


def _source_video(entry) -> Path:
    preview = Path(entry.preview_path)
    original = Path(entry.video_path)
    if original.exists() and "msrvtt_2fps_224" not in str(original):
        return original
    if preview.exists() and "msrvtt_2fps_224" not in str(preview):
        return preview
    if original.exists():
        return original
    raise HTTPException(404, f"File missing: {entry.video_path}")


def _resolve_source_path(video_id: str) -> Path:
    """Find a playable file even if the in-memory index was lost after restart."""
    idx: EmbeddingIndex = _state["index"]
    entry = idx.entry_by_id(video_id)
    if entry is not None:
        return _source_video(entry)

    search_pairs: list[tuple[str, str]] = []
    if _state.get("model_id") and _state.get("dataset_id"):
        search_pairs.append((_state["model_id"], _state["dataset_id"]))
    for model in MODELS:
        for dataset in DATASETS:
            search_pairs.append((model, dataset))

    seen: set[tuple[str, str]] = set()
    for model_id, dataset in search_pairs:
        key = (model_id, dataset)
        if key in seen:
            continue
        seen.add(key)
        _, meta = load_feature_cache(model_id, dataset)
        m = meta.get(video_id)
        if not m:
            continue
        for candidate in (m.get("preview_path"), m.get("video_path")):
            if not candidate:
                continue
            path = Path(candidate)
            if path.exists() and "msrvtt_2fps_224" not in str(path):
                return path
            if path.exists():
                return path
    raise HTTPException(404, f"Unknown video (not in current session or cache): {video_id}")


def _safe_token(text: str, fallback: str = "clip") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return (cleaned[:60] or fallback)


@app.post("/api/record")
def api_record(req: RecordRequest):
    """Export top results as a 3×3 transparent-gap GIF (for Keynote)."""
    ids = [v for v in req.video_ids if v][:9]
    if not ids:
        raise HTTPException(400, "No videos to record")

    paths: list[Path] = []
    for vid in ids:
        paths.append(_resolve_source_path(vid))

    clips_dir = CACHE_ROOT / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"results_{_safe_token(req.query or 'query')}_{stamp}.gif"
    out = clips_dir / name

    try:
        build_gif(
            paths,
            out,
            cell_w=320,
            cell_h=200,
            gap=8,
            duration=float(req.duration),
            fps=8,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"GIF export failed: {exc}") from exc

    return FileResponse(out, media_type="image/gif", filename=out.name)


@app.get("/")
def page_home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/search")
def page_search():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
