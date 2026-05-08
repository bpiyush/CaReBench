import argparse
import json
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from natsort import natsorted

STEP0_METRICS_PATH = "/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115/metrics/metrics_tarsier2_7b_nuanced_retrieval_data-validation-v1.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot validation retrieval metric and modality gap over checkpoints."
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k-stepwise",
        help="Checkpoint root containing checkpoint-*/merged_checkpoint/{embs,metrics}.",
    )
    parser.add_argument(
        "--modgap_emb_name",
        type=str,
        default="tarsier2+tara_nuanced_retrieval_data-v1_msrvtt_embeddings.pt",
        help="MSRVTT embedding filename in each checkpoint's embs directory.",
    )
    parser.add_argument(
        "--metrics_name",
        type=str,
        default="metrics_None_nuanced_retrieval_data-validation-v1.json",
        help="Metrics filename in each checkpoint's metrics directory.",
    )
    parser.add_argument(
        "--msrvtt_data_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/MSRVTT",
        help="MSRVTT data root containing annotation/msrvtt_test_1k.json.",
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        default="retrieval_and_modgap_over_steps.pdf",
        help="Output plot path.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="retrieval_vs_modgap_validation.csv",
        help="Optional output CSV path with per-checkpoint values.",
    )
    parser.add_argument(
        "--out_scatter",
        type=str,
        default="retrieval_vs_modgap_scatter.pdf",
        help="Output path for scatter plot with linear fit.",
    )
    parser.add_argument(
        "--exclude_steps",
        type=int,
        nargs="*",
        default=[10, 20],
        help="Checkpoint steps to exclude from analysis (default: 10 20).",
    )
    parser.add_argument(
        "--steps_legend_loc",
        type=str,
        default="upper left",
        help="Legend location for the step-wise plot (matplotlib loc string).",
    )
    parser.add_argument(
        "--step0_modgap",
        type=float,
        default=0.49,
        help="Step-0 modality gap value to prepend in the step-wise plot.",
    )
    return parser.parse_args()


def _normalize_stack(vectors):
    return torch.nn.functional.normalize(torch.stack(vectors), dim=-1)


def split_modal_embeddings(embs, text_keys_set):
    video_embeds = {}
    text_embeds = {}
    for key, value in embs.items():
        if key in text_keys_set:
            text_embeds[key] = value
        else:
            video_embeds[key] = value
    return video_embeds, text_embeds


def aligned_modal_tensors(df, video_embeds, text_embeds, text_col):
    video_vecs = []
    text_vecs = []
    for _, row in df.iterrows():
        video_key = str(row.get("video_id", row.get("video")))
        text_key = str(row[text_col])
        if video_key in video_embeds and text_key in text_embeds:
            video_vecs.append(video_embeds[video_key])
            text_vecs.append(text_embeds[text_key])
    if not video_vecs or not text_vecs:
        raise RuntimeError("No aligned MSRVTT video/text embeddings found for modality-gap computation.")
    X = _normalize_stack(video_vecs)
    Y = _normalize_stack(text_vecs)
    return float((X.mean(dim=0) - Y.mean(dim=0)).norm(dim=-1).item())


def average_retrieval_r1(metrics):
    if "time_v2t-ssv2" in metrics:
        bucket = metrics["time_v2t-ssv2"]
    elif "time_v2t" in metrics:
        bucket = metrics["time_v2t"]
    else:
        raise KeyError("Expected one of: 'time_v2t-ssv2' or 'time_v2t' in metrics JSON.")
    values = [
        bucket["chiral"]["R@1"],
        bucket["static"]["R@1"],
        bucket["all"]["R@1"],
    ]
    return float(np.mean(values))


def make_plot(df, out_plot, legend_loc, step0_retrieval_r1, step0_modgap):
    gap_steps = np.concatenate(([0.0], df["step"].values.astype(float)))
    gap_vals = np.concatenate(([step0_modgap], df["modality_gap"].values.astype(float)))
    corr_gap_step = np.corrcoef(gap_steps, gap_vals)[0, 1]
    retrieval_steps = np.concatenate(([0.0], df["step"].values.astype(float)))
    retrieval_vals = np.concatenate(([step0_retrieval_r1], df["retrieval_r1_avg"].values.astype(float)))
    corr_r1_step = np.corrcoef(retrieval_steps, retrieval_vals)[0, 1]

    plt.rcParams.update({"font.family": "serif"})
    fig, ax_left = plt.subplots(figsize=(7.0, 4.8))
    ax_right = ax_left.twinx()

    line_gap, = ax_left.plot(
        gap_steps,
        gap_vals,
        "-o",
        lw=2.8,
        color="indianred",
        markerfacecolor="white",
        label="Modality Gap",
    )
    retrieval_color = "seagreen"
    line_r1, = ax_right.plot(
        retrieval_steps,
        retrieval_vals,
        "-s",
        lw=2.8,
        color=retrieval_color,
        markerfacecolor="white",
        label="Avg Retrieval R@1",
    )

    ax_left.set_xlabel("Checkpoint Step")
    ax_left.set_ylabel("Modality Gap", color="indianred")
    ax_right.set_ylabel("Avg Retrieval R@1 (time_v2t-ssv2)", color=retrieval_color)
    ax_left.tick_params(axis="y", labelcolor="indianred")
    ax_right.tick_params(axis="y", labelcolor=retrieval_color)
    ax_left.set_title(
        "Modality Gap and Retrieval over Training "
        f"($\\rho_{{step,gap}}$={corr_gap_step:.2f}, $\\rho_{{step,R@1}}$={corr_r1_step:.2f})"
    )
    ax_left.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_scatter_with_linear_fit(df, out_scatter, step0_retrieval_r1, step0_modgap):
    df = pd.concat(
        [
            pd.DataFrame(
                [{"step": 0, "modality_gap": step0_modgap, "retrieval_r1_avg": step0_retrieval_r1}]
            ),
            df,
        ],
        ignore_index=True,
    )
    if len(df) < 2:
        raise RuntimeError("Need at least 2 points for scatter linear fit.")

    x = df["modality_gap"].values.astype(float)
    y = df["retrieval_r1_avg"].values.astype(float)

    plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    ax.scatter(x, y, c="tab:purple", alpha=0.85, s=45, label="Checkpoints")

    x_line = np.linspace(x.min(), x.max(), 200)
    coeffs = np.polyfit(x, y, 1)
    y_line = coeffs[0] * x_line + coeffs[1]
    ax.plot(x_line, y_line, "--", color="black", lw=3, label="Linear fit")

    corr = np.corrcoef(x, y)[0, 1]
    ax.set_xlabel("Modality Gap")
    ax.set_ylabel("Avg Retrieval R@1 (time_v2t-ssv2)")
    ax.set_title(f"Retrieval vs Modality Gap ($\\rho$={corr:.2f})")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_scatter, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    msrvtt_data = json.load(
        open(f"{args.msrvtt_data_root}/annotation/msrvtt_test_1k.json", "r", encoding="utf-8")
    )
    df_msrvtt = pd.DataFrame(msrvtt_data)
    text_col = "caption"
    text_keys_set = set(df_msrvtt[text_col].astype(str).tolist())

    emb_files = natsorted(
        glob(f"{args.ckpt_dir}/checkpoint-*/merged_checkpoint/embs/{args.modgap_emb_name}")
    )
    if not emb_files:
        raise FileNotFoundError(f"No embedding files found for pattern: {args.modgap_emb_name}")

    step_pattern = re.compile(r"checkpoint-(\d+)")
    rows = []
    for emb_path in emb_files:
        match = step_pattern.search(emb_path)
        if match is None:
            continue
        step = int(match.group(1))

        metrics_path = re.sub(
            r"/embs/[^/]+$",
            f"/metrics/{args.metrics_name}",
            emb_path,
        )
        if not glob(metrics_path):
            print(f"Skipping step {step}: missing metrics file {metrics_path}")
            continue

        embs = torch.load(emb_path, map_location="cpu")
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        video_embeds, text_embeds = split_modal_embeddings(embs, text_keys_set)
        gap = aligned_modal_tensors(df_msrvtt, video_embeds, text_embeds, text_col)
        r1_avg = average_retrieval_r1(metrics)
        rows.append({"step": step, "modality_gap": gap, "retrieval_r1_avg": r1_avg})
        print(f"Step {step}: gap={gap:.4f}, avg_R@1={r1_avg:.4f}")

    if not rows:
        raise RuntimeError("No checkpoints processed. Check input paths and filenames.")

    out_df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    with open(STEP0_METRICS_PATH, "r", encoding="utf-8") as f:
        step0_metrics = json.load(f)
    step0_retrieval_r1 = average_retrieval_r1(step0_metrics)
    print(f"Step 0: avg_R@1={step0_retrieval_r1:.4f} (from {STEP0_METRICS_PATH})")

    scatter_df = out_df.copy()
    if args.exclude_steps:
        exclude_set = set(args.exclude_steps)
        scatter_df = scatter_df[~scatter_df["step"].isin(exclude_set)].reset_index(drop=True)
        print(f"Excluded from scatter only: {sorted(exclude_set)}")

    if scatter_df.empty:
        raise RuntimeError("No checkpoints left for scatter after applying --exclude_steps.")
    out_df.to_csv(args.out_csv, index=False)
    make_plot(
        out_df,
        args.out_plot,
        args.steps_legend_loc,
        step0_retrieval_r1,
        args.step0_modgap,
    )
    make_scatter_with_linear_fit(scatter_df, args.out_scatter, step0_retrieval_r1, args.step0_modgap)
    print(f"Wrote CSV: {args.out_csv}")
    print(f"Wrote plot: {args.out_plot}")
    print(f"Wrote scatter plot: {args.out_scatter}")


if __name__ == "__main__":
    main()
