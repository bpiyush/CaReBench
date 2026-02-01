"""Test loading SigLIP model and processor."""
# Disable progress bars during download to avoid ContextVar 'shell_parent' LookupError
# (e.g. when run in certain interactive/conda contexts)
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384")
print("Loaded model and processor successfully.")
