---
license: apache-2.0
datasets:
- sentence-transformers/all-nli
language:
- en
metrics:
- accuracy
base_model:
- omni-research/Tarsier-7b
tags:
- video-retrieval
- text-to-video-retrieval
- time-awareness
- video-models
---

# ![](assets/tara-logo.png) TARA: Time-Aware Retrieval Adaptation for Video Understanding
<!-- # <img src="./assets/logo.png" width="24"> TARA: Time-Aware Retrieval Adaptation for Video Understanding -->

TARA (Time-Aware Retrieval Adaptation) is a multimodal model for video and text understanding.

## Installation & Setup

### 1. Install Git LFS (if not already installed)

Git LFS is required to download the model weights.

Please install Git LFS from https://git-lfs.github.com/.
You can refer to [this guide](https://gist.github.com/pourmand1376/bc48a407f781d6decae316a5cfa7d8ab) for non-sudo installation.
I have not tested this guide, but it should work.

Check the installation:
```bash
git lfs --version
git lfs install
```
The output should be:
```
git-lfs/3.4.1 (GitHub; linux amd64; go 1.20.11; git 0898dcbc
Updated Git hooks.
Git LFS initialized.
```


### 2. Clone the Repository
```bash
git clone https://huggingface.co/bpiyush/TARA
cd TARA
```

This will download all model weights (may take a few minutes depending on your connection).

### 3. Install Dependencies


* Create/activate the conda env (skip if you already have it):
   ```bash
   conda create -n tara python=3.10 -y
   conda activate tara
   ```
* Install CUDA 12.1 PyTorch wheels (adjust the index URL if you need a different CUDA/CPU build):
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu121 \
     torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
   ```
* Install the remaining model dependencies:
   ```bash
   pip install -r requirements.txt
   ```
* (Optional) Verify the install:
   ```bash
   python -c "import torch, transformers; print(torch.cuda.is_available(), transformers.__version__)"
   ```


## Quick Start

See the script at [demo_usage.py](demo_usage.py) for a quick start. You can run it:

```sh
python demo_usage.py
```

OR use the snippet below:

```python
import torch
from modeling_tara import TARA, read_frames_decord

model = TARA.from_pretrained(
    ".",  # Load from current directory
    device_map='auto',
    torch_dtype=torch.bfloat16,
)
n_params = sum(p.numel() for p in model.model.parameters())
print(f"Number of parameters: {round(n_params/1e9, 3)}B")

# Embed a video
video_path = "./assets/folding_paper.mp4"
video_tensor = read_frames_decord(video_path, num_frames=16)
video_tensor = video_tensor.unsqueeze(0)
video_tensor = video_tensor.to(model.model.device)
with torch.no_grad():
    video_emb = model.encode_vision(video_tensor).cpu().squeeze(0).float()
print(f"Video shape: {video_tensor.shape}")  # torch.Size([1, 16, 3, 240, 426])
print(f"Video embedding shape: {video_emb.shape}")  # torch.Size([4096])

# Embed a text
text = ['someone is folding a paper', 'cutting a paper', 'someone is folding a paper']
with torch.no_grad():
    text_emb = model.encode_text(text).cpu().float()
print(f"Text embedding shape: {text_emb.shape}")  # torch.Size([3, 4096])
```

## Citation

If you use this model, please cite:
```bibtex
@misc{tara2025,
  title={TARA: Simple and Efficient Time Aware Retrieval Adaptation of MLLMs for Video Understanding},
  author={Piyush Bagad and Andrew Zisserman},
  year={2025}
}
```

## License

Apache 2.0