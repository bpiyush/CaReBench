import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm


REQUIRED_COLUMNS = ["sent0", "sent1", "hard_neg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify temporal hard negatives using Gemini Flash."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("/scratch/shared/beegfs/piyush/datasets/SimCSE-NLI/covr/chiral10k-covr10k.csv"),
        help="Input CSV containing sent0/sent1/hard_neg columns.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("mllm4emb/temporal_hard_negative_verification_gemini.csv"),
        help="Output CSV with per-row Gemini judgements.",
    )
    parser.add_argument(
        "--prompt-path",
        type=Path,
        default=Path("mllm4emb/chiral_hard_negative_prompt.txt"),
        help="Prompt template path.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="",
        help="Optional source filter (e.g. ego4d). Leave empty to run on entire CSV.",
    )
    parser.add_argument(
        "--model-key",
        type=str,
        default="gemini-3-flash-preview",
        help="Gemini model key used by GeminiWrapper.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Checkpoint frequency in rows.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run on only 10 rows and save to CSV.",
    )
    return parser.parse_args()


def load_gemini_wrapper():
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from utils.gemini_utils import GeminiWrapper  # pylint: disable=import-outside-toplevel

    return GeminiWrapper


def validate_input_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")


def read_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def fill_prompt(template: str, row: pd.Series) -> str:
    prompt = template
    for key in REQUIRED_COLUMNS:
        prompt = prompt.replace(f"{{{key}}}", str(row[key]))
    return prompt


def parse_model_response(raw: str) -> Dict[str, object]:
    clean = re.sub(r"```json|```", "", raw or "").strip()
    parsed = json.loads(clean)
    return {
        "static_score": parsed.get("static_score"),
        "static_reason": parsed.get("static_reason"),
        "chiral_score": parsed.get("chiral_score"),
        "chiral_reason": parsed.get("chiral_reason"),
        "verdict": parsed.get("verdict"),
        "error": None,
        "raw": raw,
    }


def load_existing_output(output_csv: Path) -> pd.DataFrame:
    if output_csv.exists():
        return pd.read_csv(output_csv)
    return pd.DataFrame()


def get_completed_ids(existing_df: pd.DataFrame) -> set:
    if existing_df.empty or "row_id" not in existing_df.columns:
        return set()
    return set(existing_df["row_id"].dropna().astype(int).tolist())


def save_output(output_csv: Path, all_rows: List[Dict[str, object]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(output_csv, index=False)


def print_stats(df_out: pd.DataFrame) -> None:
    print("\n=== Verification Stats ===")
    total = len(df_out)
    print(f"total_rows: {total}")
    if total == 0:
        return

    n_errors = int(df_out["error"].notna().sum()) if "error" in df_out.columns else 0
    n_with_verdict = int(df_out["verdict"].notna().sum()) if "verdict" in df_out.columns else 0

    print(f"rows_with_verdict: {n_with_verdict}")
    print(f"rows_with_error: {n_errors}")

    if "verdict" in df_out.columns:
        verdict_counts = df_out["verdict"].fillna("MISSING").value_counts(dropna=False)
        print("\nverdict_counts:")
        for key, value in verdict_counts.items():
            print(f"  {key}: {int(value)}")

        pass_count = int((df_out["verdict"] == "PASS").sum())
        fail_count = int((df_out["verdict"] == "FAIL").sum())
        denom = max(pass_count + fail_count, 1)
        print(f"pass_rate_over_scored_rows: {pass_count / denom:.4f}")
        print(f"fail_rate_over_scored_rows: {fail_count / denom:.4f}")

    for score_col in ["static_score", "chiral_score"]:
        if score_col in df_out.columns:
            score_counts = df_out[score_col].fillna("MISSING").value_counts(dropna=False)
            print(f"\n{score_col}_counts:")
            for key, value in score_counts.items():
                print(f"  {key}: {int(value)}")


def main() -> None:
    args = parse_args()
    GeminiWrapper = load_gemini_wrapper()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    validate_input_columns(df)

    if args.source:
        if "source" not in df.columns:
            raise ValueError("`--source` provided but input CSV has no `source` column.")
        df = df[df["source"] == args.source].copy()

    if args.debug:
        df = df.sample(n=min(10, len(df)), random_state=42).copy()
        print(f"Debug mode enabled: evaluating {len(df)} samples.")

    df = df.copy()
    df["row_id"] = df.index.astype(int)

    prompt_template = read_prompt(args.prompt_path)
    vlm = GeminiWrapper(model_key=args.model_key, fps=1.0)

    existing_df = load_existing_output(args.output_csv)
    existing_rows = existing_df.to_dict("records") if not existing_df.empty else []
    completed_ids = get_completed_ids(existing_df)

    pending_df = df[~df["row_id"].isin(completed_ids)].copy()
    print(
        f"Loaded {len(df)} rows | already computed: {len(completed_ids)} | pending: {len(pending_df)}"
    )

    all_rows: List[Dict[str, object]] = existing_rows[:]
    unsaved_since = 0

    for _, row in tqdm(pending_df.iterrows(), total=len(pending_df), desc="Judging hard negatives"):
        prompt = fill_prompt(prompt_template, row)
        raw = None
        try:
            raw = vlm.forward_text_only(prompt)
            result = parse_model_response(raw)
        except Exception as exc:  # pylint: disable=broad-except
            result = {
                "static_score": None,
                "static_reason": None,
                "chiral_score": None,
                "chiral_reason": None,
                "verdict": None,
                "error": str(exc),
                "raw": raw,
            }

        out_row = {
            "row_id": int(row["row_id"]),
            "source": row.get("source", None),
            "sent0": row["sent0"],
            "sent1": row["sent1"],
            "hard_neg": row["hard_neg"],
            **result,
        }
        all_rows.append(out_row)
        unsaved_since += 1

        if unsaved_since >= args.save_every:
            save_output(args.output_csv, all_rows)
            unsaved_since = 0

    save_output(args.output_csv, all_rows)
    print(f"\nSaved results to: {args.output_csv}")

    final_df = pd.read_csv(args.output_csv)
    print_stats(final_df)


if __name__ == "__main__":
    main()
