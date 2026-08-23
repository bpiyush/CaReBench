"""Paths and constants for the video search demo."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
LOGS_DIR = ROOT / "logs"
PREVIEWS_DIR = ROOT / "cache" / "previews"

# Persistent embedding cache (dict video_id -> tensor), shared across runs
EXPERIMENTS_DIR = Path("/work/piyush/experiments/tara-presentation")
EMBEDDINGS_DIR = EXPERIMENTS_DIR / "embeddings"

for _d in (LOGS_DIR, PREVIEWS_DIR, EMBEDDINGS_DIR, STATIC_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path("/scratch/shared/beegfs/piyush/datasets")
CIA_ROOT = DATA_ROOT / "chirality-in-action"
MSRVTT_ROOT = DATA_ROOT / "MSRVTT"

TARA_PATH = Path("/work/piyush/pretrained_checkpoints/TARA")
QWEN_PATH = Path("/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B")
CAREBENCH_ROOT = Path("/users/piyush/projects/CaReBench")

QWEN_PYTHON = Path("/users/piyush/miniconda3/envs/qwen/bin/python")
CAREBENCH_PYTHON = Path("/users/piyush/miniconda3/envs/carebench/bin/python")

MODELS = {
    "tara": {
        "id": "tara",
        "label": "TARA",
        "path": str(TARA_PATH),
    },
    "qwen3vl": {
        "id": "qwen3vl",
        "label": "Qwen3VL-Embedding-8B",
        "path": str(QWEN_PATH),
    },
}

DATASETS = {
    "msrvtt": {
        "id": "msrvtt",
        "label": "MSRVTT",
        "splits": ["train", "val"],
    },
    "cia": {
        "id": "cia",
        "label": "Chirality in Action (CiA)",
        "splits": [
            "SSv2-Train",
            "SSv2-Validation",
            "EPIC-Train",
            "EPIC-Validation",
            "Charades-Train",
            "Charades-Validation",
        ],
    },
}

CIA_SOURCES = {
    "SSv2-Train": ("ssv2", "train"),
    "SSv2-Validation": ("ssv2", "validation"),
    "EPIC-Train": ("epic", "train"),
    "EPIC-Validation": ("epic", "validation"),
    "Charades-Train": ("charades", "train"),
    "Charades-Validation": ("charades", "validation"),
}

VIDEO_DIRS = {
    "ssv2": DATA_ROOT / "SSv2" / "20bn-something-something-v2",
    "epic": DATA_ROOT / "EPIC-Kitchens-100" / "cut_clips",
    "charades": DATA_ROOT / "Charades" / "Charades_v1_480_cut_clips",
}

MSRVTT_VIDEO_DIR = MSRVTT_ROOT / "videos" / "all"
MSRVTT_PREVIEW_DIR = MSRVTT_ROOT / "videos" / "msrvtt_2fps_224"
MSRVTT_TRAIN_LIST = MSRVTT_ROOT / "videos" / "train_list_new.txt"
MSRVTT_VAL_LIST = MSRVTT_ROOT / "videos" / "test_list_new.txt"
MSRVTT_ANNOTATIONS = MSRVTT_ROOT / "annotation" / "MSR_VTT.json"

PREVIEW_WIDTH = 480
QWEN_NFRAMES = 16
TOP_K_DEFAULT = 12
DEFAULT_SAMPLE_PCT = 10.0
DEFAULT_SAMPLE_SEED = 42
EXAMPLE_QUERIES = [
    "someone is folding a paper",
    "a person opening a door",
    "moving something up",
    "putting down a plate",
]
