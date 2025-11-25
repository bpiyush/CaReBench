---
license: apache-2.0
base_model: meta-llama/Llama-3.1-8B
tags:
  - video-understanding
  - multimodal
  - vision-language
  - video-encoding
  - text-encoding
  - feature-extraction
  - video-text-alignment
library_name: transformers
pipeline_tag: feature-extraction
language:
  - en
datasets:
  - nli
  - ego4d
metrics:
  - embedding-quality
  - video-text-alignment
---

# TARA Model

TARA (Time-Aware Retrieval Adaptation) is a multimodal model for video and text understanding. It can encode both videos and text into a shared embedding space, enabling tasks like video-text retrieval, video understanding, and cross-modal alignment.

## Model Details

- **Base Model**: Tarsier-7B (based on Llama-3.1-8B)
- **Architecture**: Multimodal encoder with vision and language components
- **Supported Modalities**: Video, Images, Text
- **Max Video Frames**: 32 frames

## Installation

See `INSTALL.md` for detailed installation instructions.

## Quick Start

```python
import torch
from modeling_tara import TARA

# Load the model
model = TARA.from_pretrained(
    "bpiyush/TARA",
    device_map='auto',
    torch_dtype=torch.bfloat16,
)

# Encode a video
from modeling_tara import read_frames_decord
video_tensor = read_frames_decord("path/to/video.mp4", num_frames=16)
video_tensor = video_tensor.unsqueeze(0).to(model.model.device)
with torch.no_grad():
    video_emb = model.encode_vision(video_tensor)

# Encode text
text = "someone is folding a paper"
with torch.no_grad():
    text_emb = model.encode_text(text)
```

## Usage

The model provides two main encoding methods:

- `encode_vision()`: Encodes video or image inputs into embeddings
- `encode_text()`: Encodes text inputs into embeddings

Both methods return embeddings in a shared space, enabling cross-modal tasks.

## Citation

If you use this model, please cite the original Tarsier work and this implementation.

## License

Apache 2.0

