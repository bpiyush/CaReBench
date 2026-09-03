"""JSON-line protocol worker for Qwen3VL-Embedding (runs in qwen conda env)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CAREBENCH_ROOT = Path("/users/piyush/projects/CaReBench")
PRESENTATION_ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(PRESENTATION_ROOT))
sys.path.insert(0, str(CAREBENCH_ROOT))

from config import QWEN_NFRAMES, QWEN_PATH  # noqa: E402
from models.qwen3vl_embedding import Qwen3VLEmbedder  # noqa: E402


def _reply(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _normalize(emb: torch.Tensor) -> torch.Tensor:
    return F.normalize(emb.float().cpu(), p=2, dim=-1)


def main() -> None:
    model: Qwen3VLEmbedder | None = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req.get("cmd")

        try:
            if cmd == "load":
                if model is None:
                    n_gpu = torch.cuda.device_count()
                    max_memory = {i: "14GiB" for i in range(n_gpu)}
                    max_memory["cpu"] = "64GiB"
                    model = Qwen3VLEmbedder(
                        model_name_or_path=req.get("model_path", str(QWEN_PATH)),
                        torch_dtype=torch.float16,
                        attn_implementation="flash_attention_2",
                        device_map="auto",
                        max_memory=max_memory,
                    )
                _reply({"status": "ok"})

            elif cmd == "encode_video":
                assert model is not None
                emb = model.process(
                    [{"video": req["path"], "nframes": req.get("nframes", QWEN_NFRAMES)}]
                )
                _reply({"embedding": _normalize(emb.squeeze(0)).tolist()})

            elif cmd == "encode_text":
                assert model is not None
                emb = model.process([{"text": req["text"]}])
                _reply({"embedding": _normalize(emb.squeeze(0)).tolist()})

            elif cmd == "shutdown":
                _reply({"status": "bye"})
                break

            else:
                _reply({"error": f"unknown cmd: {cmd}"})

        except Exception as exc:  # noqa: BLE001
            _reply({"error": str(exc)})


if __name__ == "__main__":
    main()
