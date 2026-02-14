"""
Evaluates Charades TestOfTime temporal understanding benchmark.

For each sample:
1. Load 8 frames from event1_clip (first/earlier clip)
2. Load 8 frames from event2_clip (second/later clip)
3. Concatenate to get 16 frames total
4. Compute video embedding zv
5. Compute text embeddings for caption (zt+) and distractor (zt-)
6. Check if sim(zv, zt+) > sim(zv, zt-)

Usage:
    python eval_charades_testoftime.py --model_path /path/to/checkpoint [--debug]
"""
import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import json
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shared.utils as su
from notebooks.eval_care_retrieval import load_model
from utils.video import read_frames_decord


# Default paths
DEFAULT_MODEL_PATH = "/work/piyush/pretrained_checkpoints/Tarsier-7b"
DEFAULT_CSV_PATH = os.path.join(os.path.dirname(__file__), "charades_testoftime.csv")
DEFAULT_CLIP_DIR = "/scratch/shared/beegfs/piyush/datasets/Charades/Charades_v1_480_cut_clips"


def load_clips_and_concatenate(
    clip1_path: str,
    clip2_path: str,
    n_frames_per_clip: int = 8,
) -> torch.Tensor:
    """
    Load frames from two separate clip files and concatenate them.
    
    Args:
        clip1_path: Path to the first clip file.
        clip2_path: Path to the second clip file.
        n_frames_per_clip: Number of frames to sample from each clip.
        
    Returns:
        Tensor of shape (n_frames_per_clip * 2, C, H, W) containing frames
        from both clips concatenated.
    """
    # Load frames from clip 1 (entire clip)
    frames_clip1 = read_frames_decord(
        clip1_path,
        num_frames=n_frames_per_clip,
    )
    
    # Load frames from clip 2 (entire clip)
    frames_clip2 = read_frames_decord(
        clip2_path,
        num_frames=n_frames_per_clip,
    )
    
    # Concatenate frames from both clips: [clip1_frames, clip2_frames]
    # Shape: (16, C, H, W)
    combined_frames = torch.cat([frames_clip1, frames_clip2], dim=0)
    
    return combined_frames


def convert_caption_to_sequence_of_events(caption):
    if " before " in caption:
        e1, e2 = caption.split(" before ")
        return f"1. {e1} \n2. {e2}"
    elif " after " in caption:
        e2, e1 = caption.split(" after ")
        return f"1. {e1} \n2. {e2}"
    else:
        raise ValueError(f"Invalid caption: {caption}")


def evaluate_charades_testoftime(
    model_path: str,
    csv_path: str,
    clip_dir: str,
    n_frames_per_clip: int = 8,
    debug: bool = False,
    device_map: str = 'auto',
    use_sequence_of_events: bool = False,
):
    """
    Evaluate a model on the Charades TestOfTime benchmark.
    
    Args:
        model_path: Path to the model checkpoint.
        csv_path: Path to the Charades TestOfTime CSV file.
        clip_dir: Directory containing the clip files.
        n_frames_per_clip: Number of frames to sample from each clip.
        debug: If True, only evaluate on a subset of samples.
        device_map: Device mapping for the model.
        use_sequence_of_events: If True, use a sequence of events as the input text.

    Returns:
        Dictionary with evaluation results.
    """
    # Load model
    print(f"Loading model from: {model_path}")
    vfc, tfc, vp = load_model(_id=model_path, device_map=device_map, n_frames=n_frames_per_clip * 2)
    
    # Load dataset
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")
    
    # Add clip paths
    df['clip1_path'] = df['event1_clip'].apply(lambda x: os.path.join(clip_dir, x))
    df['clip2_path'] = df['event2_clip'].apply(lambda x: os.path.join(clip_dir, x))
    
    # Filter to only clips that exist
    df['clip1_exists'] = df['clip1_path'].apply(os.path.exists)
    df['clip2_exists'] = df['clip2_path'].apply(os.path.exists)
    df['clips_exist'] = df['clip1_exists'] & df['clip2_exists']
    
    missing_clips = len(df[~df['clips_exist']])
    if missing_clips > 0:
        print(f"Warning: {missing_clips} samples have missing clip files")
    
    df = df[df['clips_exist']].reset_index(drop=True)
    print(f"Samples with valid clips: {len(df)}")
    
    if debug:
        # Sample a small subset for debugging
        np.random.seed(42)
        df = df.sample(min(50, len(df)), random_state=42).reset_index(drop=True)
        print(f"Debug mode: evaluating on {len(df)} samples")
    
    # Evaluate each sample
    results = []
    correct = 0
    total = 0
    
    for i in su.log.tqdm_iterator(range(len(df)), desc='Evaluating Charades TestOfTime'):
        row = df.iloc[i]
        
        try:
            # Load frames from both clips
            frames = load_clips_and_concatenate(
                clip1_path=row['clip1_path'],
                clip2_path=row['clip2_path'],
                n_frames_per_clip=n_frames_per_clip,
            )
            
            # Compute video embedding
            zv = vfc(frames)
            zv = torch.nn.functional.normalize(zv, dim=-1)
            
            # Compute text embeddings
            caption = row['caption'] if not use_sequence_of_events else convert_caption_to_sequence_of_events(row['caption'])
            distractor = row['distractor_caption'] if not use_sequence_of_events else convert_caption_to_sequence_of_events(row['distractor_caption'])

            zt_pos = tfc(caption)
            zt_pos = torch.nn.functional.normalize(zt_pos, dim=-1)
            
            zt_neg = tfc(distractor)
            zt_neg = torch.nn.functional.normalize(zt_neg, dim=-1)
            
            # Compute similarities
            sim_pos = (zv @ zt_pos).item()
            sim_neg = (zv @ zt_neg).item()
            
            # Check if correct (positive similarity should be higher)
            is_correct = sim_pos > sim_neg
            
            results.append({
                'video_id': row['video_id'],
                'conjugate': row['conjugate'],
                'event1_clip': row['event1_clip'],
                'event2_clip': row['event2_clip'],
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
            print(f"Error processing {row['video_id']} ({row['event1_clip']}, {row['event2_clip']}): {e}")
            continue
    
    # Compute metrics
    accuracy = correct / total if total > 0 else 0.0
    print(f"\n{'='*60}")
    print(f"Charades TestOfTime Evaluation Results")
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
    parser = argparse.ArgumentParser(description='Evaluate Charades TestOfTime benchmark')
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
        help='Path to the Charades TestOfTime CSV file',
    )
    parser.add_argument(
        '--clip_dir',
        type=str,
        default=DEFAULT_CLIP_DIR,
        help='Directory containing the clip files',
    )
    parser.add_argument(
        '--n_frames_per_clip',
        type=int,
        default=4,
        help='Number of frames to sample from each clip (default: 8)',
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
    parser.add_argument(
        '--use_sequence_of_events',
        action='store_true',
        help='Use a sequence of events as the input text',
    )
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_charades_testoftime(
        model_path=args.model_path,
        csv_path=args.csv_path,
        clip_dir=args.clip_dir,
        n_frames_per_clip=args.n_frames_per_clip,
        debug=args.debug,
        device_map=args.device_map,
        use_sequence_of_events=args.use_sequence_of_events,
    )
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a clean model name for the output file
    model_name = os.path.basename(args.model_path.rstrip('/'))
    if not model_name:
        model_name = 'model'
    
    output_path = os.path.join(args.output_dir, f'charades_testoftime_{model_name}.json')
    
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
        csv_output_path = os.path.join(args.output_dir, f'charades_testoftime_{model_name}_detailed.csv')
        results_df.to_csv(csv_output_path, index=False)
        print(f"Detailed results saved to: {csv_output_path}")


if __name__ == "__main__":
    main()
