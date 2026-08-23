"""FastAPI backend + static frontend for text-to-video search."""
from __future__ import annotations

import argparse
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
    DATASETS,
    EXAMPLE_QUERIES,
    MODELS,
    STATIC_DIR,
    TOP_K_DEFAULT,
    DEFAULT_SAMPLE_PCT,
    DEFAULT_SAMPLE_SEED,
)
from embedders import create_embedder
from embeddings_store import EmbeddingIndex
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
    "job": {
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


def _set_job(**kwargs):
    with _job_lock:
        _state["job"].update(kwargs)


def _get_embedder(model_id: str, model_path: str):
    if (
        _state["embedder"] is not None
        and _state["model_id"] == model_id
        and _state["model_path"] == model_path
    ):
        return _state["embedder"]
    if _state["embedder"] is not None:
        try:
            _state["embedder"].close()
        except Exception:  # noqa: BLE001
            pass
    emb = create_embedder(model_id, model_path)
    emb.load()
    _state["embedder"] = emb
    _state["model_id"] = model_id
    _state["model_path"] = model_path
    return emb


def _run_preprocess(req: PreprocessRequest):
    model = MODELS.get(req.model_id)
    dataset = DATASETS.get(req.dataset_id)
    if model is None or dataset is None:
        _set_job(status="error", error="Unknown model or dataset", progress=0.0)
        return
    if req.split not in dataset["splits"]:
        _set_job(status="error", error=f"Unknown split: {req.split}", progress=0.0)
        return

    def progress_cb(p: float, msg: str, eta: float | None):
        _set_job(progress=float(p), message=msg, eta_sec=eta)

    try:
        _set_job(
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
        _state["index"] = index
        _get_embedder(model["id"], model["path"])
        _state["model_label"] = model["label"]
        _state["dataset_id"] = dataset["id"]
        _state["dataset_label"] = dataset["label"]
        _state["split"] = req.split
        _state["sample_pct"] = req.sample_pct
        _state["sample_seed"] = req.sample_seed
        _set_job(status="done", progress=1.0, message=f"Ready — {len(index.entries)} videos", eta_sec=0.0)
    except Exception as exc:  # noqa: BLE001
        _set_job(status="error", error=str(exc), message=str(exc), progress=0.0)


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
        "job": dict(_state["job"]),
    }


@app.post("/api/preprocess")
def api_preprocess(req: PreprocessRequest):
    if _state["job"]["status"] == "running":
        raise HTTPException(409, "Preprocessing already running")
    t = threading.Thread(target=_run_preprocess, args=(req,), daemon=True)
    t.start()
    return {"ok": True}


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
    # Never serve the 2fps MSRVTT feature extract in the UI.
    if preview.exists() and "msrvtt_2fps_224" not in str(preview):
        path = preview
    elif original.exists():
        path = original
    else:
        raise HTTPException(404, f"File missing: {entry.video_path}")
    media = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
    return FileResponse(path, media_type=media, filename=path.name)


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
