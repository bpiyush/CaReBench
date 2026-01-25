"""
Text-to-Video Search Backend using FAISS for efficient similarity search.
"""
import os
import sys
import argparse
import torch
import numpy as np
import faiss
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.modeling_encoders import AutoEncoder
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__, static_folder='static')
CORS(app)

# Global variables
encoder = None
video_features = None
video_names = None
faiss_index = None
VIDEO_DIR = None
NUM_VIDEOS_TO_RETRIEVE = 20


def load_model(model_id):
    """Load the encoder model"""
    print("Loading encoder model...")
    
    # Load model - EXACTLY as provided by user
    encoder = AutoEncoder.from_pretrained(
        model_id,
        device_map='auto',
        attn_implementation="flash_attention_2",
    )
    
    print("Model loaded successfully!")
    return encoder


def load_video_features(features_path):
    """Load pre-computed video features"""
    print(f"Loading video features from {features_path}...")
    video_feats = torch.load(features_path, map_location='cpu')
    
    # Convert to numpy arrays for FAISS
    video_names = list(video_feats.keys())
    features = np.array([video_feats[name].numpy() for name in video_names], dtype=np.float32)
    
    print(f"Loaded {len(video_names)} video features")
    return video_feats, video_names, features


def build_faiss_index(features):
    """Build FAISS index for efficient similarity search"""
    print("Building FAISS index...")
    dimension = features.shape[1]
    
    # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize features for cosine similarity
    faiss.normalize_L2(features)
    
    # Add features to index
    index.add(features)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index


def encode_text(text_query):
    """Encode text query to feature vector"""
    with torch.no_grad():
        zt = encoder.encode_text(text_query).cpu().squeeze(0).float()
        zt = torch.nn.functional.normalize(zt, dim=-1)
    return zt.numpy()


@app.route('/')
def index():
    """Serve the frontend HTML page"""
    return send_from_directory('static', 'index.html')


@app.route('/search', methods=['POST'])
def search():
    """Search for videos similar to text query"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        # Cap top_k at NUM_VIDEOS_TO_RETRIEVE to limit rendering time
        top_k = min(data.get('top_k', NUM_VIDEOS_TO_RETRIEVE), NUM_VIDEOS_TO_RETRIEVE)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        print(f"Searching for: {query}")
        
        # Encode text query
        query_feat = encode_text(query).reshape(1, -1).astype(np.float32)
        
        # Normalize query feature
        faiss.normalize_L2(query_feat)
        
        # Search in FAISS index
        similarities, indices = faiss_index.search(query_feat, top_k)
        
        # Prepare results
        results = []
        for idx, sim in zip(indices[0], similarities[0]):
            results.append({
                'video_name': video_names[idx],
                'similarity': float(sim)
            })
        
        print(f"Found {len(results)} results")
        
        return jsonify({
            'query': query,
            'results': results
        })
    
    except Exception as e:
        import traceback
        error_msg = f"Search error: {str(e)}"
        print(error_msg)
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500


@app.route('/video/<path:filename>')
def serve_video(filename):
    """Serve video files"""
    return send_from_directory(VIDEO_DIR, filename)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'num_videos': len(video_names) if video_names else 0
    })


def main():
    global encoder, video_features, video_names, faiss_index, VIDEO_DIR
    
    parser = argparse.ArgumentParser(description="Text-to-Video Search Backend")
    parser.add_argument(
        "--model_id",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/"
                "nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint",
        help="Path to the encoder model"
    )
    parser.add_argument(
        "--features_path",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/CondensedMovies/outputs/features-tara7b-n=8/all.pt",
        help="Path to pre-computed video features"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/CondensedMovies/shots/",
        help="Directory containing video files"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on"
    )
    
    args = parser.parse_args()
    VIDEO_DIR = args.video_dir
    
    # Load model
    encoder = load_model(args.model_id)
    
    # Load video features
    video_features, video_names, features = load_video_features(args.features_path)
    
    # Build FAISS index
    faiss_index = build_faiss_index(features)
    
    print(f"\nStarting server on {args.host}:{args.port}")
    print(f"Ready to search {len(video_names)} videos!\n")
    
    # Run Flask app
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()

