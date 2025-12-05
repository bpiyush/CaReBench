"""
Quick startup script to test if backend can start without loading the full model.
Use this to debug path and dependency issues.
"""
import os
import sys
from pathlib import Path

print("=" * 60)
print("Checking Search Demo Setup")
print("=" * 60)

# Check paths
sys.path.append(str(Path(__file__).parent.parent))

print("\n1. Checking Python version...")
print(f"   Python: {sys.version}")

print("\n2. Checking required packages...")
try:
    import flask
    print(f"   ✓ Flask: {flask.__version__}")
except ImportError as e:
    print(f"   ✗ Flask: Not installed")
    print(f"      Install: pip install flask")

try:
    import flask_cors
    print(f"   ✓ Flask-CORS: installed")
except ImportError as e:
    print(f"   ✗ Flask-CORS: Not installed")
    print(f"      Install: pip install flask-cors")

try:
    import numpy
    print(f"   ✓ NumPy: {numpy.__version__}")
except ImportError as e:
    print(f"   ✗ NumPy: Not installed")
    print(f"      Install: pip install numpy")

try:
    import torch
    print(f"   ✓ PyTorch: {torch.__version__}")
    print(f"      CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"      GPU: {torch.cuda.get_device_name()}")
except ImportError as e:
    print(f"   ✗ PyTorch: Not installed")
    print(f"      Install: pip install torch")

try:
    import faiss
    print(f"   ✓ FAISS: installed")
except ImportError as e:
    print(f"   ✗ FAISS: Not installed")
    print(f"      Install: pip install faiss-cpu")

print("\n3. Checking model imports...")
try:
    from models.modeling_encoders import AutoEncoder
    print(f"   ✓ AutoEncoder: Can be imported")
except ImportError as e:
    print(f"   ✗ AutoEncoder: Import failed")
    print(f"      Error: {e}")

print("\n4. Checking data paths...")
model_id = "/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint"
features_path = "/scratch/shared/beegfs/piyush/datasets/CondensedMovies/outputs/features-tara7b-n=8/all.pt"
video_dir = "/scratch/shared/beegfs/piyush/datasets/CondensedMovies/shots/"

if os.path.exists(model_id):
    print(f"   ✓ Model directory exists")
else:
    print(f"   ✗ Model directory NOT FOUND")
    print(f"      Path: {model_id}")

if os.path.exists(features_path):
    print(f"   ✓ Features file exists")
    # Try to load it
    try:
        import torch
        feats = torch.load(features_path, map_location='cpu')
        print(f"      - Contains {len(feats)} video features")
        first_key = list(feats.keys())[0]
        print(f"      - Sample key: {first_key}")
        print(f"      - Feature shape: {feats[first_key].shape}")
    except Exception as e:
        print(f"      ✗ Failed to load: {e}")
else:
    print(f"   ✗ Features file NOT FOUND")
    print(f"      Path: {features_path}")

if os.path.exists(video_dir):
    print(f"   ✓ Video directory exists")
    # Count videos
    try:
        video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
        print(f"      - Contains {len(video_files)} .mp4 files")
    except Exception as e:
        print(f"      ✗ Failed to list directory: {e}")
else:
    print(f"   ✗ Video directory NOT FOUND")
    print(f"      Path: {video_dir}")

print("\n" + "=" * 60)
print("Setup check complete!")
print("=" * 60)
print("\nIf packages are missing, install them:")
print("  pip install -r requirements.txt")
print("\nTo start the backend:")
print("  python backend.py")
print("\nTo use custom paths:")
print("  python backend.py --model_id /path/to/model --features_path /path/to/features.pt")

