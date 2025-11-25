# TARA Dependencies

1. Create/activate the conda env (skip if you already have it):
   ```bash
   conda create -n tara python=3.10 -y
   conda activate tara
   ```
2. Install CUDA 12.1 PyTorch wheels (adjust the index URL if you need a different CUDA/CPU build):
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu121 \
     torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121
   ```
3. Install the remaining model dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Verify the install:
   ```bash
   python -c "import torch, transformers; print(torch.cuda.is_available(), transformers.__version__)"
   ```

