"""
Evaluates TestOfTime (TEMPO) temporal understanding benchmark.

For each video:
1. Extract 8 frames from event1 (before/first event)
2. Extract 8 frames from event2 (after/second event)
3. Concatenate to get 16 frames total
4. Compute video embedding zv
5. Compute text embeddings for caption (zt+) and distractor (zt-)
6. Check if sim(zv, zt+) > sim(zv, zt-)

Usage:
    python eval_testoftime.py --model_path /path/to/checkpoint [--debug]
"""
import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import json
from torch.nn.functional import cosine_similarity
import argparse
import PIL, PIL.Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shared.utils as su
from notebooks.eval_care_retrieval import load_model
from utils.video import read_frames_decord
from shared.utils.visualize import concat_images_with_border


# Default paths
DEFAULT_MODEL_PATH = "/work/piyush/pretrained_checkpoints/Tarsier-7b"
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "tempotl+didemo_test.csv")
DEFAULT_VIDEO_DIR = "/scratch/shared/beegfs/piyush/datasets/TestOfTime/TEMPO/videos"


def load_temporal_frames(
    video_path: str,
    event1_start: float,
    event1_end: float,
    event2_start: float,
    event2_end: float,
    n_frames_per_event: int = 8,
) -> torch.Tensor:
    """
    Load frames from two temporal events and concatenate them.
    
    Args:
        video_path: Path to the video file.
        event1_start: Start time of event 1 in seconds.
        event1_end: End time of event 1 in seconds.
        event2_start: Start time of event 2 in seconds.
        event2_end: End time of event 2 in seconds.
        n_frames_per_event: Number of frames to sample from each event.
        
    Returns:
        Tensor of shape (n_frames_per_event * 2, C, H, W) containing frames
        from both events concatenated.
    """
    # Load frames from event 1
    frames_event1 = read_frames_decord(
        video_path,
        num_frames=n_frames_per_event,
        start=event1_start,
        end=event1_end,
    )
    
    # Load frames from event 2
    frames_event2 = read_frames_decord(
        video_path,
        num_frames=n_frames_per_event,
        start=event2_start,
        end=event2_end,
    )
    
    # Concatenate frames from both events: [event1_frames, event2_frames]
    # Shape: (16, C, H, W)
    combined_frames = torch.cat([frames_event1, frames_event2], dim=0)
    
    return combined_frames


def evaluate_testoftime(
    model_path: str,
    csv_path: str,
    video_dir: str,
    n_frames_per_event: int = 8,
    debug: bool = False,
    device_map: str = 'auto',
):
    """
    Evaluate a model on the TestOfTime (TEMPO) benchmark.
    
    Args:
        model_path: Path to the model checkpoint.
        csv_path: Path to the TEMPO test CSV file.
        video_dir: Directory containing the video files.
        n_frames_per_event: Number of frames to sample from each event.
        debug: If True, only evaluate on a subset of samples.
        device_map: Device mapping for the model.
        
    Returns:
        Dictionary with evaluation results.
    """
    # Load model
    print(f"Loading model from: {model_path}")
    vfc, tfc, vp = load_model(_id=model_path, device_map=device_map, n_frames=n_frames_per_event * 2)
    
    # Load dataset
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    # Add video paths
    df['video_path'] = df['video_id'].apply(lambda x: os.path.join(video_dir, x))
    
    # Filter to only videos that exist
    df['video_exists'] = df['video_path'].apply(os.path.exists)
    missing_videos = df[~df['video_exists']]['video_id'].unique()
    if len(missing_videos) > 0:
        print(f"Warning: {len(missing_videos)} unique videos not found")
        if len(missing_videos) <= 10:
            print(f"Missing: {missing_videos}")
    
    df = df[df['video_exists']].reset_index(drop=True)
    print(f"Samples with valid videos: {len(df)}")
    
    if debug:
        # Sample a small subset for debugging
        np.random.seed(42)
        df = df.sample(min(50, len(df)), random_state=42).reset_index(drop=True)
        print(f"Debug mode: evaluating on {len(df)} samples")
    
    # Evaluate each sample
    results = []
    correct = 0
    total = 0
    
    for i in su.log.tqdm_iterator(range(len(df)), desc='Evaluating TestOfTime'):
        row = df.iloc[i]
        import ipdb; ipdb.set_trace()
        
        try:
            # Load frames from both events
            frames = load_temporal_frames(
                video_path=row['video_path'],
                event1_start=row['event1_start'],
                event1_end=row['event1_end'],
                event2_start=row['event2_start'],
                event2_end=row['event2_end'],
                n_frames_per_event=n_frames_per_event,
            )
            
            # Compute video embedding
            # Note: vp is the video processor, but we already have processed frames
            # We need to pass the frames directly through vfc
            zv = vfc(frames)
            zv = torch.nn.functional.normalize(zv, dim=-1)
            
            # Compute text embeddings
            zt_pos = tfc(row['caption'])
            zt_pos = torch.nn.functional.normalize(zt_pos, dim=-1)
            
            zt_neg = tfc(row['distractor_caption'])
            zt_neg = torch.nn.functional.normalize(zt_neg, dim=-1)
            
            # Compute similarities
            sim_pos = (zv @ zt_pos).item()
            sim_neg = (zv @ zt_neg).item()
            
            # Check if correct (positive similarity should be higher)
            is_correct = sim_pos > sim_neg
            
            results.append({
                'video_id': row['video_id'],
                'conjugate': row['conjugate'],
                'caption': row['caption'],
                'distractor_caption': row['distractor_caption'],
                'sim_positive': sim_pos,
                'sim_negative': sim_neg,
                'correct': is_correct,
            })
            
            if is_correct:
                correct += 1
            total += 1
            
        except Exception as e:
            print(f"Error processing {row['video_id']}: {e}")
            continue
    
    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"TestOfTime Evaluation Results")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Total samples evaluated: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"{'='*60}")
    
    # Breakdown by conjugate type (before/after)
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        print("\nAccuracy by conjugate type:")
        for conjugate in results_df['conjugate'].unique():
            subset = results_df[results_df['conjugate'] == conjugate]
            conj_acc = subset['correct'].mean() * 100
            print(f"  {conjugate}: {conj_acc:.2f}% ({subset['correct'].sum()}/{len(subset)})")
    
    return {
        'accuracy': accuracy,
        'total': total,
        'correct': correct,
        'model_path': model_path,
        'results': results_df.to_dict('records') if len(results_df) > 0 else [],
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate TestOfTime (TEMPO) benchmark')
    parser.add_argument(
        '--model_path',
        type=str,
        default=DEFAULT_MODEL_PATH,
        help='Path to the model checkpoint',
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default=DEFAULT_CSV_PATH,
        help='Path to the TEMPO test CSV file',
    )
    parser.add_argument(
        '--video_dir',
        type=str,
        default=DEFAULT_VIDEO_DIR,
        help='Directory containing the video files',
    )
    parser.add_argument(
        '--n_frames_per_event',
        type=int,
        default=8,
        help='Number of frames to sample from each event (default: 8)',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode with subset of samples',
    )
    parser.add_argument(
        '--device_map',
        type=str,
        default='auto',
        help='Device mapping for model loading',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Directory to save results',
    )
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_testoftime(
        model_path=args.model_path,
        csv_path=args.csv_path,
        video_dir=args.video_dir,
        n_frames_per_event=args.n_frames_per_event,
        debug=args.debug,
        device_map=args.device_map,
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a clean model name for the output file
    model_name = os.path.basename(args.model_path.rstrip('/'))
    if not model_name:
        model_name = 'model'
    
    output_path = os.path.join(args.output_dir, f'testoftime_{model_name}.json')
    
    # Save summary (without full results to keep file size manageable)
    summary = {
        'accuracy': results['accuracy'],
        'total': results['total'],
        'correct': results['correct'],
        'model_path': results['model_path'],
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    # Also save detailed results as CSV
    if results['results']:
        results_df = pd.DataFrame(results['results'])
        csv_output_path = os.path.join(args.output_dir, f'testoftime_{model_name}_detailed.csv')
        results_df.to_csv(csv_output_path, index=False)
        print(f"Detailed results saved to: {csv_output_path}")


if __name__ == "__main__":
    main()
