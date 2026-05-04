import argparse
import re
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from natsort import natsorted

import shared.utils as su


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure modality gap over training and optionally save t-SNE GIF."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/scratch/shared/beegfs/piyush/datasets/MSRVTT",
        help="MSRVTT data root containing annotation/msrvtt_test_1k.json.",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="/work/piyush/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k-stepwise",
        help="Checkpoint root containing checkpoint-*/merged_checkpoint/embs/*.pt files.",
    )
    parser.add_argument(
        "--emb_glob",
        type=str,
        default="tarsier2+tara_nuanced_retrieval_data-v1_msrvtt_embeddings.pt",
        help="Embedding filename pattern within each checkpoint embs dir.",
    )
    parser.add_argument(
        "--out_plot",
        type=str,
        default="modgap_over_training.pdf",
        help="Output path for loss/modality-gap plot.",
    )
    parser.add_argument(
        "--gif",
        action="store_true",
        help="If set, save a slow GIF of t-SNE modality-gap evolution over time.",
    )
    parser.add_argument(
        "--out_gif",
        type=str,
        default="modgap_tsne_over_training.gif",
        help="Output path for t-SNE GIF (used only when --gif is passed).",
    )
    parser.add_argument(
        "--gif_frame_duration",
        type=float,
        default=1.0,
        help="Frame duration in seconds for GIF (default: 1 step per second).",
    )
    parser.add_argument(
        "--tsne_perplexity",
        type=float,
        default=30.0,
        help="t-SNE perplexity passed to shared.utils.visualize.reduce_dim.",
    )
    return parser.parse_args()


def _normalize_stack(vectors):
    return torch.nn.functional.normalize(torch.stack(vectors), dim=-1)


def aligned_modal_tensors(df, video_embeds, text_embeds, text_col):
    X = []
    Y = []
    for _, row in df.iterrows():
        video_key = row.get("video_id", row.get("video"))
        text_key = row[text_col]
        if video_key in video_embeds and text_key in text_embeds:
            X.append(video_embeds[video_key])
            Y.append(text_embeds[text_key])
    if not X or not Y:
        raise RuntimeError("No aligned video/text embeddings were found for modality-gap computation.")
    return _normalize_stack(X), _normalize_stack(Y)


def modality_gap_from_tensors(X, Y):
    return np.round((X.mean(dim=0) - Y.mean(dim=0)).norm(dim=-1).item(), 2)


def split_modal_embeddings(embs, text_keys_set):
    video_embeds = {}
    text_embeds = {}
    for key, value in embs.items():
        if key in text_keys_set:
            text_embeds[key] = value
        else:
            video_embeds[key] = value
    return video_embeds, text_embeds


def make_dual_axis_plot(steps, losses, deltas, out_path):
    loss_color = "royalblue"
    gap_color = "indianred"

    import scipy.stats

    correlation, _ = scipy.stats.pearsonr(losses[: len(steps)], deltas[: len(steps)])
    print(f"Pearson correlation between loss and delta: {correlation}")

    plt.rcParams.update({"font.family": "serif"})
    fig, ax_left = plt.subplots(figsize=(6, 4))
    ax_right = ax_left.twinx()

    line_loss, = ax_left.plot(
        steps,
        losses[: len(steps)],
        "-o",
        color=loss_color,
        lw=2,
        label="Loss",
        markerfacecolor="none",
    )
    line_gap, = ax_right.plot(
        steps,
        deltas,
        "-s",
        color=gap_color,
        lw=2,
        label="Modality Gap",
        alpha=0.7,
        markerfacecolor="none",
    )

    ax_left.set_xlabel("Step")
    ax_left.set_ylabel("Loss", color=loss_color)
    ax_right.set_ylabel("Modality Gap", color=gap_color)
    ax_left.tick_params(axis="y", labelcolor=loss_color)
    ax_right.tick_params(axis="y", labelcolor=gap_color)
    ax_left.set_title(f"Modality gap and loss over training ($ \\rho = {np.round(correlation, 2)} $)")
    ax_left.grid(alpha=0.25)
    ax_left.legend([line_loss, line_gap], ["Loss", "Modality Gap"], loc="best")

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_tsne_gif(frames_data, out_gif, perplexity, frame_duration):
    import imageio.v2 as imageio

    if not frames_data:
        print("No frames available for GIF; skipping.")
        return

    reduced_frames = []
    xmins, xmaxs, ymins, ymaxs = [], [], [], []
    for frame in frames_data:
        combined = np.concatenate([frame["video"], frame["text"]], axis=0)
        Z = su.visualize.reduce_dim(combined, method="tsne", perplexity=perplexity)
        n_video = frame["video"].shape[0]
        zv = Z[:n_video]
        zt = Z[n_video:]
        reduced_frames.append((frame["step"], frame["delta"], zv, zt))
        xmins.append(Z[:, 0].min())
        xmaxs.append(Z[:, 0].max())
        ymins.append(Z[:, 1].min())
        ymaxs.append(Z[:, 1].max())

    xpad = 0.05 * (max(xmaxs) - min(xmins) + 1e-9)
    ypad = 0.05 * (max(ymaxs) - min(ymins) + 1e-9)
    xlim = (min(xmins) - xpad, max(xmaxs) + xpad)
    ylim = (min(ymins) - ypad, max(ymaxs) + ypad)

    images = []
    for step, delta, zv, zt in reduced_frames:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(zv[:, 0], zv[:, 1], s=8, c="tab:blue", alpha=0.85, label="Video")
        ax.scatter(zt[:, 0], zt[:, 1], s=8, c="tab:orange", alpha=0.85, label="Text")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        ax.set_xlabel(r"$z_1$")
        ax.set_ylabel(r"$z_2$")
        ax.set_title(f"Step {step}    " + r"$\Delta_{gap}$" + f" = {delta:.2f}")
        ax.legend(loc="upper right")
        plt.tight_layout()

        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
        images.append(img)
        plt.close(fig)

    imageio.mimsave(out_gif, images, duration=frame_duration, loop=0)
    print(f"Wrote t-SNE GIF: {out_gif}")


def main():
    args = parse_args()
    data_root = args.data_root
    video_dir = f"{data_root}/videos/all/"
    text_col = "caption"

    data = su.io.load_json(f"{data_root}/annotation/msrvtt_test_1k.json")
    df = pd.DataFrame(data)
    df["video_path"] = df.video.apply(lambda x: f"{video_dir}/{x}")

    embed_files = natsorted(
        glob(f"{args.ckpt_dir}/checkpoint-*/merged_checkpoint/embs/{args.emb_glob}")
    )
    if not embed_files:
        raise FileNotFoundError(
            f"No embedding files found in {args.ckpt_dir} matching pattern {args.emb_glob}"
        )

    step_pattern = re.compile(r"checkpoint-(\d+)")
    step_count = []
    for path in embed_files:
        match = step_pattern.search(path)
        if match is None:
            raise ValueError(f"Could not parse checkpoint step from path: {path}")
        step_count.append(int(match.group(1)))

    text_keys_set = set(df[text_col].tolist())

    steps = [0]
    deltas = [0.49]
    tsne_frames = []
    for emb_path, step in zip(embed_files, step_count):
        embs = torch.load(emb_path, map_location="cpu")
        video_embeds, text_embeds = split_modal_embeddings(embs, text_keys_set)
        X, Y = aligned_modal_tensors(df, video_embeds, text_embeds, text_col)
        delta = modality_gap_from_tensors(X, Y)
        print(f"Step {step}: Modality gap = {delta}")
        steps.append(step)
        deltas.append(delta)
        if args.gif:
            tsne_frames.append(
                {
                    "step": step,
                    "delta": float(delta),
                    "video": X.numpy(),
                    "text": Y.numpy(),
                }
            )

    losses = [
        6.43,
        4.50,
        0.56,
        0.24,
        0.29,
        0.30,
        0.33,
        0.36,
        0.23,
        0.23,
        0.15,
        0.24,
        0.27,
        0.17,
    ]
    all_steps = np.arange(0, 140, 10)
    losses = [losses[i] for i in range(len(all_steps)) if all_steps[i] in steps]

    make_dual_axis_plot(steps, losses, deltas, args.out_plot)
    print(f"Wrote plot: {args.out_plot}")

    if args.gif:
        save_tsne_gif(
            tsne_frames,
            out_gif=args.out_gif,
            perplexity=args.tsne_perplexity,
            frame_duration=args.gif_frame_duration,
        )


if __name__ == "__main__":
    main()