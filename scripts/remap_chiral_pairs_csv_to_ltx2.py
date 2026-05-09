#!/usr/bin/env python3
"""
Remap video paths in a chiral-pairs CSV from Wan (or any prior root) to LTX2 layout.

LTX2 layout (under <ltx2_root>):
  videos/  — {source_index:06d}_{slug}_{hash}.mp4 and the same stem + -reversed.mp4
  metadata/ — one JSON per forward clip; field "source_index" matches CSV row order

Captions (sent0, sent_hard_neg) are copied unchanged; only video0 / video_hard_neg paths change.

By default only rows where BOTH mp4 files exist are written (--filter-incomplete).
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Optional, Tuple

import fire


def _resolve_pair(videos_dir: Path, source_index: int) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (forward_mp4, reversed_mp4) or (None, None) if forward is missing or ambiguous."""
    pat = f"{source_index:06d}_*.mp4"
    candidates = sorted(
        p
        for p in videos_dir.glob(pat)
        if p.is_file() and not p.name.endswith("-reversed.mp4")
    )
    if len(candidates) != 1:
        return None, None
    forward = candidates[0]
    reversed_p = forward.with_name(forward.stem + "-reversed.mp4")
    if not reversed_p.is_file():
        return forward, None
    return forward, reversed_p


def remap(
    input_csv: str = "data/generated-chiral-pairs-v2.csv",
    output_csv: str = "data/generated-chiral-pairs-v2-ltx2-complete.csv",
    ltx2_root: str = "/scratch/shared/beegfs/piyush/datasets/LTX2/LTX2_full_81f_480x320_8gpu",
    filter_incomplete: bool = True,
) -> None:
    in_path = Path(input_csv)
    out_path = Path(output_csv)
    root = Path(ltx2_root)
    videos_dir = root / "videos"
    if not videos_dir.is_dir():
        raise FileNotFoundError(f"Missing LTX2 videos directory: {videos_dir}")

    with in_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames: List[str] = reader.fieldnames or []
        if not fieldnames or not all(
            k in fieldnames for k in ("sent0", "sent_hard_neg", "video0", "video_hard_neg")
        ):
            raise ValueError(f"Unexpected CSV columns in {in_path}: {fieldnames}")
        rows = list(reader)

    written = 0
    skipped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, row in enumerate(rows):
            fwd, rev = _resolve_pair(videos_dir, i)
            if fwd is None or rev is None:
                skipped += 1
                if not filter_incomplete:
                    raise RuntimeError(
                        f"Row {i}: missing LTX2 pair under {videos_dir} "
                        f"(forward_ok={fwd is not None}, reversed_ok={rev is not None}). "
                        "Use filter_incomplete=True or finish generating reversed videos."
                    )
                continue
            new_row = dict(row)
            new_row["video0"] = str(fwd.resolve())
            new_row["video_hard_neg"] = str(rev.resolve())
            writer.writerow(new_row)
            written += 1

    print(f"Input rows: {len(rows)}")
    print(f"LTX2 root: {root}")
    print(f"Wrote: {out_path} ({written} rows)")
    if filter_incomplete:
        print(f"Skipped (incomplete pair): {skipped}")


if __name__ == "__main__":
    fire.Fire(remap)
