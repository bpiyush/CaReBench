"""
Label NLI triplets for whether the hard negative is based on negation of the anchor.
Uses Qwen3-4B to classify each triplet and outputs structured JSON labels.
"""

import argparse
import pandas as pd
import json
import os
import sys
import re
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils.qwen3_utils import load_model, generate_answers_batch

import shared.utils as su


PROMPT_TEMPLATE = """\
You are a linguistic analyst. Given an NLI triplet (anchor, positive, hard_negative), determine if the hard_negative contradicts the anchor using **explicit negation words**.

STEP 1: Scan the hard_negative for negation words/morphemes: "not", "n't" (isn't, doesn't, won't, can't, couldn't, hasn't, weren't, etc.), "no", "none", "never", "nobody", "nothing", "nowhere", "neither", "nor".
STEP 2: If any negation word is found AND it is used to deny/negate a claim from the anchor, then is_negation = true. Otherwise false.

CRITICAL RULES:
- ONLY explicit negation words count. Antonyms (e.g., "smiled" vs "frowned"), replacements (e.g., "red brick" vs "gray marble"), restrictors ("only"), or opposite adjectives ("slow" vs "fast") are NOT negation.
- The negation word must appear in the hard_negative sentence itself.
- Even if negation words are embedded in a longer sentence with other content, they still count.

Examples of NEGATION (is_negation = true):
- Anchor: "Ask for a bedroom facing the Old City walls." | Negative: "The Old City walls do not exist." | negation_words: ["not"] ✓
- Anchor: "We sought to identify practices." | Negative: "We don't want to identify any agent practices." | negation_words: ["don't"] ✓
- Anchor: "There will be a child abuse campaign." | Negative: "Nobody is organizing a child abuse campaign." | negation_words: ["nobody"] ✓
- Anchor: "The woman went to find the father." | Negative: "The woman did not care where the man was." | negation_words: ["not"] ✓

Examples of NON-NEGATION (is_negation = false):
- Anchor: "Place des Vosges has stone and red brick facades." | Negative: "Place des Vosges is constructed entirely of gray marble." | negation_words: [] (material replacement)
- Anchor: "She smiled back." | Negative: "She frowned." | negation_words: [] (antonym)
- Anchor: "My walkman broke." | Negative: "My walkman still works." | negation_words: [] (contrary assertion)
- Anchor: "Fun for adults and children." | Negative: "Fun for only children." | negation_words: [] (restrictor, not negation)

Now classify this triplet:

Anchor: {anchor}
Positive: {positive}
Hard Negative: {negative}

Respond ONLY with a JSON object (no other text):
{{"is_negation": true/false, "negation_words": ["list", "of", "found", "words"] or [], "reason": "brief explanation in 1 sentence"}}"""


def build_prompt(anchor, positive, negative):
    return PROMPT_TEMPLATE.format(
        anchor=anchor, positive=positive, negative=negative,
    )


def parse_response(text):
    """Extract JSON from model response, handling common formatting issues."""
    text = text.strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON from markdown code block
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding first JSON object in text
    match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"is_negation": None, "reason": f"PARSE_ERROR: {text[:200]}"}


def main():
    parser = argparse.ArgumentParser(
        description="Label NLI triplets for negation-based hard negatives using Qwen3-4B."
    )
    parser.add_argument(
        "--csv_path", type=str, required=True,
        help="Path to the NLI CSV file with columns: sent0, sent1, hard_neg",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Run on only 50 samples for quick testing",
    )
    parser.add_argument(
        "--si", type=int, default=None,
        help="Start index (inclusive) for processing a slice of the CSV",
    )
    parser.add_argument(
        "--ei", type=int, default=None,
        help="End index (exclusive) for processing a slice of the CSV",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size for model inference (default: 16)",
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=192,
        help="Max new tokens per response (default: 192)",
    )
    parser.add_argument(
        "--output_dir", type=str,
        default="/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/outputs",
        help="Directory to save output files",
    )
    parser.add_argument(
        "--save_every", type=int, default=5000,
        help="Save checkpoint every N samples (default: 5000)",
    )
    parser.add_argument(
        "--resume_from", type=str, default=None,
        help="Path to a partial output JSONL to resume from",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    su.log.print_update("Loading CSV")
    df = pd.read_csv(args.csv_path)
    total = len(df)
    su.log.print_update(f"Loaded {total} triplets from {args.csv_path}")

    if args.debug:
        df = df.head(50)
        su.log.print_update("DEBUG mode: using first 50 samples")

    # Apply start/end index slicing
    si = args.si if args.si is not None else 0
    ei = args.ei if args.ei is not None else len(df)
    si = max(0, si)
    ei = min(len(df), ei)
    df = df.iloc[si:ei].reset_index(drop=True)
    su.log.print_update(f"Sliced to indices [{si}, {ei}): {len(df)} samples")

    # Handle resume
    start_idx = 0
    existing_results = []
    if args.resume_from and os.path.exists(args.resume_from):
        su.log.print_update(f"Resuming from {args.resume_from}")
        with open(args.resume_from, "r") as f:
            for line in f:
                existing_results.append(json.loads(line))
        start_idx = len(existing_results)
        su.log.print_update(f"Loaded {start_idx} existing results, resuming from index {start_idx}")

    df = df.iloc[start_idx:]
    if len(df) == 0:
        su.log.print_update("All samples already processed. Exiting.")
        return

    su.log.print_update(f"Processing {len(df)} remaining samples (indices {si + start_idx} to {si + start_idx + len(df) - 1})")

    # Build prompts
    prompts = []
    for _, row in df.iterrows():
        prompts.append(build_prompt(
            anchor=str(row["sent0"]),
            positive=str(row["sent1"]),
            negative=str(row["hard_neg"]),
        ))

    # Load model
    su.log.print_update("Loading Qwen3-4B model")
    model, tokenizer = load_model(
        model_name="/work/piyush/pretrained_checkpoints/Qwen3-4B-Instruct-2507/"
    )

    # Output file path
    suffix = "_debug" if args.debug else ""
    if args.si is not None or args.ei is not None:
        suffix += f"_{si}-{ei}"
    output_path = os.path.join(args.output_dir, f"negation_labels{suffix}.jsonl")
    if args.resume_from:
        output_path = args.resume_from

    # Write existing results if starting fresh
    if not args.resume_from:
        open(output_path, "w").close()

    su.log.print_update(f"Output will be saved to {output_path}")

    # Process in chunks for periodic saving
    results = list(existing_results)
    chunk_size = args.save_every
    n_prompts = len(prompts)

    for chunk_start in range(0, n_prompts, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_prompts)
        chunk_prompts = prompts[chunk_start:chunk_end]
        global_start = si + start_idx + chunk_start

        su.log.print_update(
            f"Processing chunk [{global_start} - {global_start + len(chunk_prompts) - 1}] "
            f"({chunk_start + len(chunk_prompts)}/{n_prompts} done)"
        )

        t0 = time.time()
        responses = generate_answers_batch(
            model, tokenizer, chunk_prompts,
            max_new_tokens=args.max_new_tokens,
            enable_thinking=False,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0
        su.log.print_update(
            f"Chunk generated in {elapsed:.1f}s "
            f"({elapsed / len(chunk_prompts):.2f}s per sample)"
        )

        chunk_results = []
        df_chunk = df.iloc[chunk_start:chunk_end]
        for i, (resp, (_, row)) in enumerate(zip(responses, df_chunk.iterrows())):
            parsed = parse_response(resp["content"])
            result = {
                "index": global_start + i,
                "sent0": str(row["sent0"]),
                "sent1": str(row["sent1"]),
                "hard_neg": str(row["hard_neg"]),
                "is_negation": parsed.get("is_negation"),
                "negation_words": parsed.get("negation_words", []),
                "reason": parsed.get("reason", ""),
                "raw_response": resp["content"],
            }
            chunk_results.append(result)

        results.extend(chunk_results)

        with open(output_path, "a") as f:
            for r in chunk_results:
                f.write(json.dumps(r) + "\n")

        su.log.print_update(f"Saved checkpoint: {len(results)} total results to {output_path}")

    # Summary statistics
    n_negation = sum(1 for r in results if r.get("is_negation") is True)
    n_non_negation = sum(1 for r in results if r.get("is_negation") is False)
    n_parse_error = sum(1 for r in results if r.get("is_negation") is None)
    su.log.print_update(
        f"Done! Total: {len(results)} | "
        f"Negation: {n_negation} ({100*n_negation/len(results):.1f}%) | "
        f"Non-negation: {n_non_negation} ({100*n_non_negation/len(results):.1f}%) | "
        f"Parse errors: {n_parse_error}"
    )


if __name__ == "__main__":
    main()
