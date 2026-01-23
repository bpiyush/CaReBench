"""
Script to create a processed CSV from TEMPO TL+DiDeMo test data.

The TEMPO dataset has annotations with "before" and "after" conjugations connecting
two events. This script processes these annotations and creates a CSV where each
row contains:
- video_id: The video file name
- conjugate: The temporal conjugate word (before/after)
- event1_description: Description of the first event (chronologically)
- event2_description: Description of the second event (chronologically)
- caption: Complete caption of format "event before/after event"
- distractor_caption: Swapped caption (wrong temporal order)
- event1_start: Start time of the first event (in seconds)
- event1_end: End time of the first event (in seconds)
- event2_start: Start time of the second event (in seconds)
- event2_end: End time of the second event (in seconds)

Temporal format:
- Each clip is 5 seconds long
- Clip IDs [start, end] map to seconds as: (start*5, (end+1)*5)
- For example: [0,0] → (0s, 5s), [1,2] → (5s, 15s)

Event ordering:
- For "before" annotations: "A before B" -> A=event1 (first), B=event2 (second)
- For "after" annotations: "A after B" -> B=event1 (first), A=event2 (second)

Distractor caption:
- Swap the event order while keeping the conjugate word
- "A before B" -> distractor: "B before A" (incorrect: B doesn't happen before A)
- "A after B" -> distractor: "B after A" (incorrect: B doesn't happen after A)
"""

import json
import pandas as pd
from pathlib import Path
from collections import Counter


def clip_to_seconds(clip_range: list) -> tuple:
    """Convert clip IDs to seconds.
    
    Each clip is 5 seconds long.
    [0, 0] → (0, 5)
    [1, 2] → (5, 15)
    [0, 2] → (0, 15)
    """
    start_clip, end_clip = clip_range
    start_sec = start_clip * 5
    end_sec = (end_clip + 1) * 5
    return start_sec, end_sec


def majority_vote_times(times_list: list) -> list:
    """Get majority vote from 4 annotators' timestamps."""
    # Convert to tuples for counting
    times_tuples = [tuple(t) for t in times_list]
    counter = Counter(times_tuples)
    majority = counter.most_common(1)[0][0]
    return list(majority)


def process_annotation(annotation: dict) -> dict:
    """Process a single annotation to extract event times and descriptions.
    
    Two formats exist for captions:
    
    Standard format:
        "A before B" -> A happens first, B happens second
        "A after B"  -> B happens first, A happens second
        
    Inverted format (starts with conjugate):
        "before B, A" or "Before B, A" -> A happens first, B happens second
        "after B, A" or "After B, A"   -> B happens first, A happens second
    
    Event ordering:
        - event1 = chronologically first event
        - event2 = chronologically second event
        
    Distractor caption: swap events to create incorrect temporal claim
    """
    annotation_id = annotation['annotation_id']
    video = annotation['video']
    description = annotation['description']
    
    # Get majority vote for times
    times_majority = majority_vote_times(annotation['times'])
    context = annotation['context']
    
    # Determine conjugate and extract event descriptions
    if annotation_id.startswith('before_'):
        conjugate = 'before'
        desc_lower = description.lower()
        
        # Check for inverted format: "before B, A" or "Before B, A"
        if desc_lower.startswith('before '):
            # Format: "before B, A" -> split by comma
            # Remove leading "before " and split by comma
            rest = description[7:]  # Remove "before " (7 chars)
            if ',' in rest:
                parts = rest.split(',', 1)
                event2_desc = parts[0].strip()  # B (second event, reference)
                event1_desc = parts[1].strip()  # A (first event)
            else:
                raise ValueError(f"Cannot parse inverted 'before' format: {description}")
        else:
            # Standard format: "A before B"
            parts = description.split(' before ', 1)
            if len(parts) != 2:
                raise ValueError(f"Cannot parse 'before' in: {description}")
            event1_desc = parts[0].strip()  # A (first event)
            event2_desc = parts[1].strip()  # B (second event, reference)
        
        event1_clips = times_majority
        event2_clips = context
        
        # Create normalized caption (always "A before B" format) and distractor
        caption = f"{event1_desc} before {event2_desc}"
        distractor_caption = f"{event2_desc} before {event1_desc}"
        
    elif annotation_id.startswith('after_'):
        conjugate = 'after'
        desc_lower = description.lower()
        
        # Check for inverted format: "after B, A" or "After B, A"
        if desc_lower.startswith('after '):
            # Format: "After B, A" -> B happens first (reference), A happens second
            rest = description[6:]  # Remove "after " (6 chars)
            if ',' in rest:
                parts = rest.split(',', 1)
                event1_desc = parts[0].strip()  # B (first event, reference)
                event2_desc = parts[1].strip()  # A (second event)
            else:
                raise ValueError(f"Cannot parse inverted 'after' format: {description}")
        else:
            # Standard format: "A after B"
            parts = description.split(' after ', 1)
            if len(parts) != 2:
                raise ValueError(f"Cannot parse 'after' in: {description}")
            event2_desc = parts[0].strip()  # A (second event)
            event1_desc = parts[1].strip()  # B (first event, reference)
        
        event1_clips = context
        event2_clips = times_majority
        
        # Create normalized caption (always "A after B" format) and distractor
        caption = f"{event2_desc} after {event1_desc}"
        distractor_caption = f"{event1_desc} after {event2_desc}"
        
    else:
        raise ValueError(f"Unknown annotation type: {annotation_id}")
    
    # Convert clips to seconds
    event1_start, event1_end = clip_to_seconds(event1_clips)
    event2_start, event2_end = clip_to_seconds(event2_clips)
    
    return {
        'video_id': video,
        'conjugate': conjugate,
        'event1_description': event1_desc,
        'event2_description': event2_desc,
        'caption': caption,
        'distractor_caption': distractor_caption,
        'event1_start': event1_start,
        'event1_end': event1_end,
        'event2_start': event2_start,
        'event2_end': event2_end,
    }


def main():
    # Paths
    data_dir = Path("/scratch/shared/beegfs/piyush/datasets/TestOfTime")
    csv_file = data_dir / "TEMPO/initial_release_data/tempoTL+didemo_test_temporal_1k.csv"
    json_file = data_dir / "TEMPO/initial_release_data/tempoTL+didemo_test.json"
    output_dir = Path("/users/piyush/projects/CaReBench/testoftime-eval")
    output_file = output_dir / "tempotl+didemo_test.csv"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load the CSV to get the list of annotation IDs to process
    csv_df = pd.read_csv(csv_file)
    print(f"Loaded {len(csv_df)} rows from CSV")
    
    # Load the JSON with all annotations
    with open(json_file) as f:
        all_annotations = json.load(f)
    print(f"Loaded {len(all_annotations)} annotations from JSON")
    
    # Create a mapping from annotation_id to annotation
    annotation_map = {ann['annotation_id']: ann for ann in all_annotations}
    
    # Process each row in the CSV
    processed_rows = []
    missing_count = 0
    
    for _, row in csv_df.iterrows():
        annotation_id = row['annotation_id']
        
        if annotation_id not in annotation_map:
            print(f"Warning: annotation_id {annotation_id} not found in JSON")
            missing_count += 1
            continue
        
        annotation = annotation_map[annotation_id]
        processed = process_annotation(annotation)
        processed_rows.append(processed)
    
    # Create DataFrame and save
    output_df = pd.DataFrame(processed_rows)
    output_df.to_csv(output_file, index=False)
    
    print(f"\nProcessed {len(processed_rows)} annotations")
    print(f"Missing annotations: {missing_count}")
    print(f"Output saved to: {output_file}")
    
    # Print summary statistics
    print(f"\nConjugate distribution:")
    print(output_df['conjugate'].value_counts())
    
    # Verify we have exactly 1000 rows
    assert len(output_df) == 1000, f"Expected 1000 rows, got {len(output_df)}"
    print(f"\nVerified: Output has exactly {len(output_df)} rows")


if __name__ == "__main__":
    main()
