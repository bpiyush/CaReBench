"""Shared CLIP / DINO.txt / SigLIP2 text and image encoders."""
from dataclasses import dataclass
from typing import Callable, List, Sequence

import torch
import torch.nn.functional as F
from PIL import Image


MODEL_DEFAULTS = {
    "clip": "ViT-L/14",
    "dinotxt": "dinov2_vitl14_reg4_dinotxt_tet1280d20h24l",
    "siglip2": "google/siglip2-so400m-patch14-384",
}


def _stage(msg: str) -> None:
    print(msg, flush=True)


@dataclass
class VLModalityEncoder:
    name: str
    model_id: str
    encode_text_batch: Callable[[Sequence[str]], torch.Tensor]
    encode_image_batch: Callable[[Sequence[str]], torch.Tensor]


def load_clip_encoder(model_id: str, device: torch.device) -> VLModalityEncoder:
    import clip

    _stage(f"[clip] Loading weights ({model_id})...")
    model, preprocess = clip.load(model_id, device=device)
    model.eval()

    def encode_text_batch(texts: Sequence[str]) -> torch.Tensor:
        tokens = clip.tokenize(list(texts), truncate=True).to(device)
        with torch.no_grad():
            z = model.encode_text(tokens).float()
            z = F.normalize(z, dim=-1)
        return z.cpu()

    def encode_image_batch(paths: Sequence[str]) -> torch.Tensor:
        images = []
        for path in paths:
            with Image.open(path) as img:
                images.append(preprocess(img.convert("RGB")))
        batch = torch.stack(images).to(device)
        with torch.no_grad():
            z = model.encode_image(batch).float()
            z = F.normalize(z, dim=-1)
        return z.cpu()

    return VLModalityEncoder("clip", model_id, encode_text_batch, encode_image_batch)


def load_dinotxt_encoder(model_id: str, device: torch.device) -> VLModalityEncoder:
    _stage(
        f"[dinotxt] torch.hub.load('facebookresearch/dinov2', '{model_id}') — "
        "may download ~1GB+ on first run and take several minutes..."
    )
    model = torch.hub.load("facebookresearch/dinov2", model_id, trust_repo=True)
    _stage(f"[dinotxt] Weights loaded; moving model to {device}...")
    model = model.to(device)
    model.eval()

    _stage("[dinotxt] Loading tokenizer and image transforms...")
    from dinov2.hub.dinotxt import get_tokenizer
    from dinov2.data.transforms import make_classification_eval_transform

    tokenizer = get_tokenizer()
    preprocess = make_classification_eval_transform()
    _stage("[dinotxt] Model setup complete.")

    def encode_text_batch(texts: Sequence[str]) -> torch.Tensor:
        tokens = tokenizer.tokenize(list(texts)).to(device)
        with torch.autocast(device.type, dtype=torch.float):
            with torch.no_grad():
                z = model.encode_text(tokens).float()
                z = F.normalize(z, dim=-1)
        return z.cpu()

    def encode_image_batch(paths: Sequence[str]) -> torch.Tensor:
        images = torch.stack(
            [preprocess(Image.open(p).convert("RGB")) for p in paths]
        ).to(device)
        with torch.autocast(device.type, dtype=torch.float):
            with torch.no_grad():
                z = model.encode_image(images).float()
                z = F.normalize(z, dim=-1)
        return z.cpu()

    return VLModalityEncoder("dinotxt", model_id, encode_text_batch, encode_image_batch)


def load_siglip2_encoder(model_id: str, device: torch.device) -> VLModalityEncoder:
    from transformers import AutoModel, AutoProcessor

    _stage(f"[siglip2] Loading processor and weights ({model_id})...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device)
    _stage("[siglip2] Model setup complete.")
    model.eval()

    def _text_inputs(texts: Sequence[str]):
        return processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )

    def encode_text_batch(texts: Sequence[str]) -> torch.Tensor:
        inputs = _text_inputs(texts)
        text_inputs = {
            k: v.to(device)
            for k, v in inputs.items()
            if k in ("input_ids", "attention_mask")
        }
        with torch.no_grad():
            z = model.get_text_features(**text_inputs).float()
            z = F.normalize(z, dim=-1)
        return z.cpu()

    def encode_image_batch(paths: Sequence[str]) -> torch.Tensor:
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            z = model.get_image_features(pixel_values=pixel_values).float()
            z = F.normalize(z, dim=-1)
        return z.cpu()

    return VLModalityEncoder("siglip2", model_id, encode_text_batch, encode_image_batch)


def load_encoder(model_name: str, model_id: str, device: torch.device) -> VLModalityEncoder:
    loaders = {
        "clip": load_clip_encoder,
        "dinotxt": load_dinotxt_encoder,
        "siglip2": load_siglip2_encoder,
    }
    if model_name not in loaders:
        raise ValueError(f"Unknown model: {model_name}")
    _stage(f"Initializing {model_name} encoder...")
    return loaders[model_name](model_id, device)
