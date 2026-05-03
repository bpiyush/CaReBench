"""
Text-to-Video Retrieval Evaluation
===================================
Computes retrieval metrics from a .pt file of precomputed embeddings.

Expected .pt format (from observed structure)
---------------------------------------------
    X = torch.load("embeddings.pt")
    X.keys()  # 'dataset_root', 'index_json', 'model_name', ..., 'entries'

    # Video entry
    X['entries'][i] = {
        'modality':   'video',
        'key':        '/path/to/video.mp4',
        'video_id':   '0000',
        'embedding':  Tensor[D],
        'video_file': 'videos/0000.mp4',
        'label':      'up',           # direction class
        'sub_label':  'up_medium',    # direction_speed
    }

    # Text entry
    X['entries'][i] = {
        'modality':   'text',
        'key':        'red circle moving up slowly',
        'video_id':   '0000',         # which video this query belongs to
        'embedding':  Tensor[D],
        'label':      'up',
        'sub_label':  'up_medium',
    }

Metrics
-------
  Overall  R@{1,5,10}, MedR, MeanR  - first retrieved video (by similarity) whose
                                       **shape** and **direction** both match the
                                       GT clip's metadata; R@K if that hit lies in
                                       top-K (not exact ``video_id`` match).
  Dynamic  R@{1,5,10}               - any of top-K shares **direction** with GT
  Static   R@{1,5,10}               - any of top-K shares **shape** with GT
                                       (needs shape in metadata, e.g. ``index.json``)

Usage
-----
    python evaluate_retrieval.py --features embeddings.pt
    python evaluate_retrieval.py --features embeddings.pt --metadata index.json
    python evaluate_retrieval.py --features embeddings.pt --inspect
    python evaluate_retrieval.py --features embeddings.pt --save_results out.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Attribute keys used for static / dynamic matching
# ---------------------------------------------------------------------------

def _parse_sub_label(sub_label: str) -> Tuple[str, str]:
    """'up_medium' -> ('up', 'medium').  Handles edge cases gracefully."""
    parts = sub_label.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return parts[0], "unknown"


def static_match(meta_a: dict, meta_b: dict) -> bool:
    """Same object shape only (both sides must have a shape string)."""
    sa, sb = meta_a.get("shape"), meta_b.get("shape")
    if sa is None or sb is None or str(sa) == "" or str(sb) == "":
        return False
    return sa == sb


def dynamic_match(meta_a: dict, meta_b: dict) -> bool:
    """Same motion direction only (e.g. up / down / left / right)."""
    return meta_a.get("direction") == meta_b.get("direction")


def overall_match(gt_meta: dict, cand_meta: dict) -> bool:
    """Shape and direction both match GT clip attributes."""
    return static_match(gt_meta, cand_meta) and dynamic_match(gt_meta, cand_meta)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

class RetrievalDataset:
    def __init__(self):
        self.queries:      List[str]       = []
        self.query_embs:   List            = []
        self.gt_video_ids: List[str]       = []

        self.video_ids:    List[str]       = []
        self.video_embs:   List            = []
        self.video_meta:   Dict[str, dict] = {}   # video_id -> metadata

    @property
    def has_static_meta(self) -> bool:
        """True if we have shape per video (from embeddings + optional index)."""
        if not self.video_meta:
            return False
        sample = next(iter(self.video_meta.values()))
        sh = sample.get("shape")
        return sh is not None and str(sh) != ""

    def video_matrix(self) -> torch.Tensor:
        return F.normalize(torch.stack(self.video_embs), dim=-1)   # [N, D]

    def query_matrix(self) -> torch.Tensor:
        return F.normalize(torch.stack(self.query_embs), dim=-1)   # [Q, D]


def _inspect(data: dict) -> None:
    print("\n-- .pt file top-level keys --")
    for k, v in data.items():
        if k == "entries":
            print(f"  'entries'  list[{len(v)}]")
            modalities = {}
            for e in v:
                m = e.get("modality", "?")
                modalities[m] = modalities.get(m, 0) + 1
            for m, cnt in modalities.items():
                print(f"    modality='{m}' : {cnt} entries")
            video_ex = next((e for e in v if e.get("modality") == "video"), None)
            text_ex  = next((e for e in v if e.get("modality") == "text"),  None)
            if video_ex:
                print(f"  Sample video entry:\n    { {k: (v.shape if hasattr(v,'shape') else v) for k,v in video_ex.items()} }")
            if text_ex:
                print(f"  Sample text entry:\n    { {k: (v.shape if hasattr(v,'shape') else v) for k,v in text_ex.items()} }")
        else:
            print(f"  {k!r}  ->  {str(v)[:100]}")
    print("-----------------------------\n")


def load_features(pt_path: str,
                  meta_path: Optional[str] = None,
                  inspect: bool = False) -> RetrievalDataset:

    print(f"Loading: {pt_path}")
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(pt_path, map_location="cpu")

    if inspect:
        _inspect(data)

    if "entries" not in data:
        sys.exit("Error: .pt file has no 'entries' key. Run --inspect to debug.")

    entries = data["entries"]

    # ── Optional sidecar metadata (index.json from generator) ─────────────
    extra_meta: Dict[str, dict] = {}
    if meta_path:
        mp = Path(meta_path)
        if not mp.is_file():
            print(f"  Warning: metadata file not found ({mp}); continuing without sidecar.")
        else:
            with open(mp) as f:
                idx = json.load(f)
            if "videos" in idx:
                extra_meta = {str(v["video_id"]): v for v in idx["videos"]}
            else:
                extra_meta = {str(k): v for k, v in idx.items() if isinstance(v, dict)}
            print(f"Loaded sidecar metadata for {len(extra_meta)} videos from {mp}")

    # ── Split by modality ──────────────────────────────────────────────────
    video_entries = [e for e in entries if e.get("modality") == "video"]
    text_entries  = [e for e in entries if e.get("modality") == "text"]

    if not video_entries:
        sys.exit("Error: no entries with modality='video' found.")
    if not text_entries:
        sys.exit("Error: no entries with modality='text' found.")

    ds = RetrievalDataset()

    # ── Build video index ──────────────────────────────────────────────────
    for ve in video_entries:
        vid = str(ve["video_id"])
        direction, speed = _parse_sub_label(ve.get("sub_label", "unknown_unknown"))
        meta = {
            "direction":   direction,
            "speed_label": speed,
            "label":       ve.get("label", direction),
            "sub_label":   ve.get("sub_label", ""),
            "video_file":  ve.get("video_file", ""),
        }
        if ve.get("shape") is not None:
            meta["shape"] = ve["shape"]
        if ve.get("obj_color_name") is not None:
            meta["obj_color_name"] = ve["obj_color_name"]
        em = extra_meta.get(vid)
        if em:
            meta.update(em)   # adds shape, speed_label, queries, etc.

        ds.video_ids.append(vid)
        ds.video_embs.append(ve["embedding"].float())
        ds.video_meta[vid] = meta

    # ── Build query index ──────────────────────────────────────────────────
    for te in text_entries:
        ds.queries.append(str(te.get("key", te.get("query", ""))))
        ds.query_embs.append(te["embedding"].float())
        ds.gt_video_ids.append(str(te["video_id"]))

    return ds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(ds: RetrievalDataset,
             ks: Tuple[int, ...] = (1, 5, 10)) -> dict:

    N = len(ds.video_ids)
    Q = len(ds.queries)

    if N == 0 or Q == 0:
        sys.exit("Dataset appears empty after loading.")

    vid2idx = {vid: i for i, vid in enumerate(ds.video_ids)}

    V  = ds.video_matrix()   # [N, D]
    Qm = ds.query_matrix()   # [Q, D]
    sim    = Qm @ V.T        # [Q, N]  cosine similarity (both L2-normalised)
    ranked = torch.argsort(sim, dim=1, descending=True)  # [Q, N]

    results = []
    skipped = 0
    for qi in range(Q):
        gt_vid = ds.gt_video_ids[qi]
        gt_idx = vid2idx.get(gt_vid)

        if gt_idx is None:
            skipped += 1
            continue

        gt_meta   = ds.video_meta[gt_vid]
        rank_list = ranked[qi].tolist()

        r1_vid  = ds.video_ids[rank_list[0]]
        r1_meta = ds.video_meta[r1_vid]

        r1_static = static_match(gt_meta, r1_meta) if ds.has_static_meta else None
        r1_dynamic = dynamic_match(gt_meta, r1_meta)
        r1_overall = overall_match(gt_meta, r1_meta) if ds.has_static_meta else None

        if ds.has_static_meta:
            overall_rank = N + 1
            for pos, idx in enumerate(rank_list):
                cand_vid = ds.video_ids[idx]
                if overall_match(gt_meta, ds.video_meta[cand_vid]):
                    overall_rank = pos + 1
                    break
            overall_at_k = {k: overall_rank <= k for k in ks}
        else:
            overall_rank = N + 1
            overall_at_k = {k: False for k in ks}

        dynamic_at_k = {}
        static_at_k = {}
        for k in ks:
            top_k_metas = [ds.video_meta[ds.video_ids[idx]] for idx in rank_list[:k]]
            dynamic_at_k[k] = any(dynamic_match(gt_meta, m) for m in top_k_metas)
            if ds.has_static_meta:
                static_at_k[k] = any(static_match(gt_meta, m) for m in top_k_metas)

        results.append({
            "query":          ds.queries[qi],
            "gt_vid":         gt_vid,
            "overall_rank":   overall_rank,
            "r1_vid":         r1_vid,
            "direction":      gt_meta.get("direction", "?"),
            "speed_label":    gt_meta.get("speed_label", "?"),
            "shape":          gt_meta.get("shape", "?"),
            "static_r1":      r1_static,
            "dynamic_r1":     r1_dynamic,
            "r1_overall":     r1_overall,
            "overall_at_k":   overall_at_k,
            "static_at_k":    static_at_k,
            "dynamic_at_k":   dynamic_at_k,
        })

    if skipped:
        print(f"  Warning: {skipped} queries skipped (GT video not in video set).")

    return {
        "results":         results,
        "N_videos":        N,
        "Q_queries":       len(results),
        "ks":              list(ks),
        "has_static_meta": ds.has_static_meta,
        "model_name":      "",
    }


# ---------------------------------------------------------------------------
# Pretty report
# ---------------------------------------------------------------------------

def _pct(hits: list) -> str:
    if not hits:
        return "  n/a "
    return f"{100 * sum(hits) / len(hits):5.1f}%"


def print_report(ev: dict) -> None:
    results    = ev["results"]
    ks         = ev["ks"]
    Q          = ev["Q_queries"]
    N          = ev["N_videos"]
    has_static = ev["has_static_meta"]

    all_ranks = [r["overall_rank"] for r in results]
    med_rank  = sorted(all_ranks)[len(all_ranks) // 2]
    mean_rank = sum(all_ranks) / len(all_ranks)

    overall_hits = {k: [r["overall_at_k"][k]  for r in results] for k in ks}
    dynamic_hits = {k: [r["dynamic_at_k"][k]  for r in results] for k in ks}
    static_hits  = {k: [r["static_at_k"].get(k, False) for r in results]
                    for k in ks} if has_static else None

    W   = 74
    SEP = "-" * W

    print()
    print("=" * W)
    print(f"{'TEXT -> VIDEO RETRIEVAL':^{W}}")
    if ev.get("model_name"):
        print(f"{'model: ' + ev['model_name']:^{W}}")
    print("=" * W)
    print(f"  Videos : {N}    Queries : {Q}    "
          f"Queries/video : {Q/N:.1f}    Dynamic : direction only")
    print(SEP)

    k_hdr = "".join(f"   R@{k:<3}" for k in ks)
    print(f"  {'Metric':<30}{k_hdr}   MedR   MeanR")
    print(SEP)

    def row(label, hits_by_k, show_rank=False, note=""):
        cols   = "".join(f"   {_pct(hits_by_k[k]):<5}" for k in ks)
        rank_s = f"   {med_rank:<6.0f} {mean_rank:<6.1f}" if show_rank else ""
        note_s = f"  <- {note}" if note else ""
        print(f"  {label:<30}{cols}{rank_s}{note_s}")

    if has_static:
        row("Overall  (shape + direction hit)", overall_hits, show_rank=True)
    else:
        print(f"  {'Overall  (shape + direction hit)':<30}"
              + "   n/a  " * len(ks)
              + "   n/a     n/a  "
              + "  <- need shape in metadata (--metadata index.json)")
    row("Dynamic  (direction)", dynamic_hits)
    if has_static:
        row("Static   (shape)", static_hits)
    else:
        print(f"  {'Static   (shape)':<30}"
              + "   n/a  " * len(ks)
              + "  <- merge index.json (--metadata) or store shape on video entries")

    # ── Per-direction breakdown ───────────────────────────────────────────
    directions = sorted({r["direction"] for r in results})
    print()
    print(SEP)
    print("  Overall (shape+dir) R@K  by DIRECTION")
    print(SEP)
    for d in directions:
        sub  = [r for r in results if r["direction"] == d]
        hits = {k: [r["overall_at_k"][k] for r in sub] for k in ks}
        n    = len(sub)
        row(f"  {d.upper():<28}", hits, note=f"n={n}")

    # ── Per-speed breakdown ───────────────────────────────────────────────
    speed_order = {"slow": 0, "medium": 1, "fast": 2}
    speeds = sorted({r["speed_label"] for r in results},
                    key=lambda s: speed_order.get(s, 99))
    print()
    print(SEP)
    print("  Overall (shape+dir) R@K  by SPEED")
    print(SEP)
    for s in speeds:
        sub  = [r for r in results if r["speed_label"] == s]
        hits = {k: [r["overall_at_k"][k] for r in sub] for k in ks}
        n    = len(sub)
        row(f"  {s:<28}", hits, note=f"n={n}")

    # ── Static x Dynamic cross-tab at R@1 ────────────────────────────────
    print()
    print(SEP)
    if has_static:
        print("  Static x Dynamic cross-tab at R@1")
        print(SEP)
        for sv in [True, False]:
            for dv in [True, False]:
                n  = sum(1 for r in results
                         if r["static_r1"] == sv and r["dynamic_r1"] == dv)
                sl = "Static OK " if sv else "Static FAIL"
                dl = "Dynamic OK " if dv else "Dynamic FAIL"
                bar = int(36 * n / max(Q, 1))
                print(f"  {sl} & {dl}: {'#'*bar} {n}  ({100*n/Q:.1f}%)")
    else:
        print("  Dynamic correctness at R@1")
        print(SEP)
        for dv in [True, False]:
            n   = sum(1 for r in results if r["dynamic_r1"] == dv)
            dl  = "Dynamic OK  " if dv else "Dynamic FAIL"
            bar = int(36 * n / max(Q, 1))
            print(f"  {dl}: {'#'*bar} {n}  ({100*n/Q:.1f}%)")

    print("=" * W)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Text-to-Video Retrieval Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--features",      required=True,
                        help="Path to .pt embeddings file")
    parser.add_argument("--metadata",      default="/scratch/shared/beegfs/piyush/datasets/SyntheticMotion/spatial_motion_dataset/index.json",
                        help="Path to index.json (enables static metrics)")
    parser.add_argument("--ks",            nargs="+", type=int, default=[1, 5, 10],
                        help="Values of K for R@K (default: 1 5 10)")
    parser.add_argument("--inspect",       action="store_true",
                        help="Print .pt structure and exit")
    parser.add_argument("--save_results",  default=None,
                        help="Save per-query results to JSON")
    args = parser.parse_args()

    if not Path(args.features).exists():
        sys.exit(f"Error: file not found: {args.features}")

    ds = load_features(args.features, args.metadata, inspect=args.inspect)

    if args.inspect:
        print(f"Videos  : {len(ds.video_ids)}")
        print(f"Queries : {len(ds.queries)}")
        if ds.video_meta:
            sample_id = ds.video_ids[0]
            print(f"\nSample video metadata ({sample_id}):")
            print(json.dumps(ds.video_meta[sample_id], indent=2, default=str))
        print(f"\nStatic meta available : {ds.has_static_meta}")
        return

    ev = evaluate(ds, ks=tuple(args.ks))

    # Grab model name from .pt if available
    try:
        try:
            raw = torch.load(args.features, map_location="cpu", weights_only=False)
        except TypeError:
            raw = torch.load(args.features, map_location="cpu")
        ev["model_name"] = raw.get("model_name", "")
    except Exception:
        pass

    print_report(ev)

    if args.save_results:
        with open(args.save_results, "w", encoding="utf-8") as f:
            json.dump(ev, f, indent=2, default=str, ensure_ascii=False)
        print(f"Saved per-query results -> {args.save_results}")


if __name__ == "__main__":
    main()