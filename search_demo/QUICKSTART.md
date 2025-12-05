# Quick Start Guide

## The Issue

You're getting "Search request failed" because the Python dependencies aren't installed yet in your current environment.

## Solution

### Step 1: Install Dependencies

```bash
cd /users/piyush/projects/CaReBench/search_demo
pip install -r requirements.txt
```

**Note:** The requirements.txt contains:
- flask
- flask-cors  
- torch
- numpy
- faiss-cpu

If you have a different conda environment or want to use specific versions already in your environment, you can install just the missing ones:

```bash
pip install flask flask-cors faiss-cpu
```

### Step 2: Verify Setup

```bash
python check_setup.py
```

This will verify:
- ✓ All packages are installed
- ✓ Model directory exists  
- ✓ Features file exists (645,958 videos!)
- ✓ Video directory exists

### Step 3: Start the Backend

```bash
python backend.py
```

This will:
1. Load the encoder model (~30-60 seconds)
2. Load pre-computed video features
3. Build FAISS index for fast search
4. Start Flask server on http://0.0.0.0:5000

### Step 4: Use the Search Demo

Open your browser and go to:
```
http://localhost:5000
```

Then:
1. Type a query like "a person walking on the beach"
2. Click Search or press Enter
3. Browse through the results!

## UI Changes

The UI now has a **clean, light theme** with:
- ✓ Light gray background (#f5f7fa)
- ✓ White cards with subtle shadows
- ✓ Blue accent color (#3498db)
- ✓ Green similarity scores
- ✓ No more purple gradient!

## Troubleshooting

### If backend won't start:
```bash
# Check what's wrong
python check_setup.py

# Make sure you're in the right environment
which python
```

### If videos won't load:
Check the browser console (F12) for errors. The videos are served from:
```
/scratch/shared/beegfs/piyush/datasets/CondensedMovies/shots/
```

### If search is slow:
Consider installing FAISS-GPU instead:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

## Custom Paths

If you need to use different paths:

```bash
python backend.py \
  --model_id /path/to/your/model \
  --features_path /path/to/features.pt \
  --video_dir /path/to/videos \
  --port 8080
```

---

**That's it! Your text-to-video search demo should now work perfectly.**

