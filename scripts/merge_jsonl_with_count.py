#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge JSONL files into one output file with row count in filename."
    )
    parser.add_argument(
        "--input-glob",
        required=True,
        help='Glob for input JSONL files, e.g. "data/ann/**/*.jsonl"',
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write merged output file.",
    )
    parser.add_argument(
        "--output-prefix",
        default="merged",
        help="Output filename prefix.",
    )
    parser.add_argument(
        "--skip-invalid-json",
        action="store_true",
        help="Skip invalid JSON lines instead of failing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_files = sorted(glob.glob(args.input_glob, recursive=True))
    if not input_files:
        raise FileNotFoundError(f"No files matched --input-glob: {args.input_glob}")

    os.makedirs(args.output_dir, exist_ok=True)

    merged_lines = []
    for path in input_files:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    if args.skip_invalid_json:
                        print(f"[warn] Skipping invalid JSON at {path}:{line_no} ({exc})")
                        continue
                    raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
                merged_lines.append(json.dumps(obj, ensure_ascii=False))

    row_count = len(merged_lines)
    output_name = f"{args.output_prefix}-rows_{row_count}.jsonl"
    output_path = Path(args.output_dir) / output_name

    with open(output_path, "w", encoding="utf-8") as out:
        for line in merged_lines:
            out.write(line + "\n")

    print(f"Merged {len(input_files)} files")
    print(f"Total rows: {row_count}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
