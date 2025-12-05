# Text-to-Video Search Demo

A modern, efficient text-to-video search system using FAISS for fast similarity search and a beautiful web interface.

## Features

- 🚀 **Fast Search**: Uses FAISS for efficient similarity search over video embeddings
- 🎨 **Modern UI**: Beautiful, responsive interface with gradient design
- 🎬 **Video Preview**: Grid view with inline video players
- 📊 **Similarity Scores**: Shows cosine similarity percentage for each result
- 📄 **Pagination**: Browse through results with 40 videos per page, 200 total results
- ⚡ **Real-time**: Instant search results with smooth animations

## Architecture

### Backend (`backend.py`)
- **Flask** web server with CORS support
- **FAISS** (Facebook AI Similarity Search) for efficient vector search
- **PyTorch** for model inference
- **AutoEncoder** model for text encoding
- Pre-computed video features loaded at startup

### Frontend (`static/index.html`)
- Pure HTML/CSS/JavaScript (no frameworks needed)
- Responsive grid layout
- Smooth animations and transitions
- Video player with controls

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you have GPU support and want to use FAISS-GPU for even faster search:
```bash
pip uninstall faiss-cpu
pip install faiss-gpu
```

### 2. Verify Data Paths

Make sure the following paths exist and are accessible:

- **Video files**: `/scratch/shared/beegfs/piyush/datasets/CondensedMovies/shots/`
- **Video features**: `/scratch/shared/beegfs/piyush/datasets/CondensedMovies/outputs/features-tara7b-n=8/all.pt`
- **Model checkpoint**: Default path in `compute_features.py`

### 3. Run the Server

```bash
cd search_demo
python backend.py
```

By default, the server runs on `http://0.0.0.0:5000`

#### Command-line Arguments

```bash
python backend.py --help
```

Available options:
- `--model_id`: Path to encoder model (default: from compute_features.py)
- `--features_path`: Path to pre-computed features (default: all.pt)
- `--video_dir`: Directory containing videos (default: shots/)
- `--port`: Server port (default: 5000)
- `--host`: Server host (default: 0.0.0.0)

Example with custom paths:
```bash
python backend.py \
    --model_id /path/to/model \
    --features_path /path/to/features.pt \
    --video_dir /path/to/videos \
    --port 8080
```

## Usage

1. **Start the server** (see above)
2. **Open browser** and navigate to `http://localhost:5000`
3. **Enter search query** in the search bar (e.g., "a person running on the beach")
4. **Click Search** or press Enter
5. **Browse results** using pagination
6. **Click videos** to play them inline

## API Endpoints

### `POST /search`
Search for videos matching a text query.

**Request:**
```json
{
  "query": "a person running",
  "top_k": 200
}
```

**Response:**
```json
{
  "query": "a person running",
  "results": [
    {
      "video_name": "ed9ae8e0d028_155.84_163.84.mp4",
      "similarity": 0.8523
    },
    ...
  ]
}
```

### `GET /video/<filename>`
Serve a video file.

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "num_videos": 50000
}
```

## Performance

- **Initial Load**: ~30-60 seconds (loading model and building FAISS index)
- **Search Time**: ~10-50ms for 50,000 videos (using FAISS-CPU)
- **Search Time**: ~5-20ms for 50,000 videos (using FAISS-GPU)

## Technical Details

### FAISS Index
- Uses `IndexFlatIP` (Inner Product) with normalized vectors for cosine similarity
- All video features are normalized to unit length
- Query features are also normalized before search

### Text Encoding
```python
with torch.no_grad():
    zt = encoder.encode_text("Sample sentence").squeeze(0).cpu().float()
    zt = torch.nn.functional.normalize(zt, dim=-1)
```

### Video Features
- Pre-computed using `compute_features.py`
- Stored as PyTorch dict: `{video_filename: torch.Tensor([4096])}`
- Normalized for cosine similarity

## Troubleshooting

### Server won't start
- Check that all paths exist
- Verify GPU availability if using flash_attention_2
- Check port is not already in use

### Videos won't play
- Verify video directory path is correct
- Check browser console for errors
- Ensure videos are in a browser-compatible format (MP4)

### Slow search
- Consider using FAISS-GPU instead of FAISS-CPU
- Check that features are properly normalized
- Monitor system resources (RAM, GPU memory)

### Model loading errors
- Verify model checkpoint path
- Check CUDA availability
- Try using `attn_implementation="sdpa"` instead of "flash_attention_2"

## Future Improvements

- [ ] Add video thumbnails for faster loading
- [ ] Implement caching for common queries
- [ ] Add filters (duration, date, etc.)
- [ ] Support for batch queries
- [ ] Add video metadata display
- [ ] Export search results

## License

This is part of the CaReBench project.

