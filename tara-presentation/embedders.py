"""Model embedding backends for TARA, Qwen3VL, and CLIP (avgpool)."""
from __future__ import annotations

import gc
import json
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn.functional as F

from config import (
    CAREBENCH_ROOT,
    CLIP_NFRAMES,
    CLIP_PATH,
    QWEN_NFRAMES,
    QWEN_PATH,
    QWEN_PYTHON,
    TARA_PATH,
)

PRESENTATION_ROOT = Path(__file__).resolve().parent


def release_gpu_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class EmbedderBackend(ABC):
    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def encode_video(self, path: str) -> torch.Tensor: ...

    @abstractmethod
    def encode_text(self, text: str) -> torch.Tensor: ...

    @abstractmethod
    def close(self) -> None: ...


class TaraEmbedder(EmbedderBackend):
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or str(TARA_PATH)
        self.model = None

    def load(self) -> None:
        if self.model is not None:
            return
        sys.path.insert(0, self.model_path)
        from modeling_tara import TARA  # type: ignore[import-not-found]

        self.model = TARA.from_pretrained(
            self.model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

    def encode_video(self, path: str) -> torch.Tensor:
        assert self.model is not None
        with torch.no_grad():
            emb = self.model.encode_vision(path).cpu().squeeze(0).float()
        return F.normalize(emb, p=2, dim=-1)

    def encode_text(self, text: str) -> torch.Tensor:
        assert self.model is not None
        with torch.no_grad():
            emb = self.model.encode_text(text).cpu().squeeze(0).float()
        return F.normalize(emb, p=2, dim=-1)

    def close(self) -> None:
        model = self.model
        self.model = None
        if model is not None:
            del model
        release_gpu_memory()


class ClipAvgPoolEmbedder(EmbedderBackend):
    """OpenAI CLIP frame features averaged over time (ZSAR CLIP avg. baseline)."""

    def __init__(self, model_path: str | None = None, nframes: int = CLIP_NFRAMES):
        self.model_path = model_path or str(CLIP_PATH)
        self.nframes = int(nframes)
        self.model = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        if self.model is not None:
            return
        import clip  # type: ignore[import-not-found]

        # clip.load accepts a model name or a local .pt checkpoint path.
        source = self.model_path
        load_arg = source if Path(source).is_file() else "ViT-L/14"
        self.model, self.preprocess = clip.load(load_arg, device=self.device)
        self.model.eval()
        if str(CAREBENCH_ROOT) not in sys.path:
            sys.path.insert(0, str(CAREBENCH_ROOT))

    def encode_video(self, path: str) -> torch.Tensor:
        assert self.model is not None and self.preprocess is not None
        from shared.utils.video import load_frames_linspace  # type: ignore[import-not-found]

        frames = load_frames_linspace(path, n=self.nframes)
        x = torch.stack([self.preprocess(f) for f in frames]).to(self.device)
        with torch.no_grad():
            z = self.model.encode_image(x).float()
            z = z.mean(dim=0).cpu()
        return F.normalize(z, p=2, dim=-1)

    def encode_text(self, text: str) -> torch.Tensor:
        assert self.model is not None
        import clip  # type: ignore[import-not-found]

        tokens = clip.tokenize([text], truncate=True).to(self.device)
        with torch.no_grad():
            z = self.model.encode_text(tokens).cpu().squeeze(0).float()
        return F.normalize(z, p=2, dim=-1)

    def close(self) -> None:
        model = self.model
        self.model = None
        self.preprocess = None
        if model is not None:
            del model
        release_gpu_memory()


class QwenSubprocessEmbedder(EmbedderBackend):
    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or str(QWEN_PATH)
        self.proc: subprocess.Popen | None = None
        self.worker_script = Path(__file__).resolve().parent / "workers" / "qwen_worker.py"
        self._stderr_log: Path | None = None

    def _stderr_text(self) -> str:
        if self._stderr_log and self._stderr_log.exists():
            return self._stderr_log.read_text(errors="replace")[-4000:]
        return ""

    def _send(self, payload: dict) -> dict:
        assert self.proc is not None and self.proc.stdin and self.proc.stdout
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(
                f"Qwen worker exited unexpectedly.\n{self._stderr_text()}"
            )
        resp = json.loads(line)
        if "error" in resp:
            extra = self._stderr_text()
            msg = resp["error"]
            if extra:
                msg = f"{msg}\n--- worker log ---\n{extra}"
            raise RuntimeError(msg)
        return resp

    def load(self) -> None:
        if self.proc is not None:
            return
        log_dir = PRESENTATION_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self._stderr_log = log_dir / "qwen_worker.log"
        stderr_f = open(self._stderr_log, "w", encoding="utf-8")  # noqa: SIM115
        env = dict(**{k: v for k, v in __import__("os").environ.items()})
        env["PYTHONPATH"] = f"{PRESENTATION_ROOT}:{CAREBENCH_ROOT}:{env.get('PYTHONPATH', '')}"
        self.proc = subprocess.Popen(
            [str(QWEN_PYTHON), str(self.worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr_f,
            text=True,
            bufsize=1,
            env=env,
            cwd=str(PRESENTATION_ROOT),
        )
        self._send({"cmd": "load", "model_path": self.model_path})

    def encode_video(self, path: str) -> torch.Tensor:
        resp = self._send({"cmd": "encode_video", "path": path, "nframes": QWEN_NFRAMES})
        return torch.tensor(resp["embedding"], dtype=torch.float32)

    def encode_text(self, text: str) -> torch.Tensor:
        resp = self._send({"cmd": "encode_text", "text": text})
        return torch.tensor(resp["embedding"], dtype=torch.float32)

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self._send({"cmd": "shutdown"})
        except Exception:  # noqa: BLE001
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None
        release_gpu_memory()


def create_embedder(model_id: str, model_path: str) -> EmbedderBackend:
    if model_id == "tara":
        return TaraEmbedder(model_path)
    if model_id == "qwen3vl":
        return QwenSubprocessEmbedder(model_path)
    if model_id == "clip":
        return ClipAvgPoolEmbedder(model_path)
    raise ValueError(f"Unknown model id: {model_id}")
