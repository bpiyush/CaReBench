import os

import torch
import pandas as pd

import shared.utils as su


def video_edit_eol_prompt():
    prompt = "Source video: <video>\nEdit instruction: <sent>\n"\
    "Look at the attached video carefully. The provided text is instruction to edit the video. "\
    "Imagine this edit instruction being applied to the provided video frame.\n"\
    "Summarize the resulting edited video in one word:"
    prompt = f"USER: {prompt} ASSISTANT: "
    return prompt


data_dir = {
    "cia-ssv2": "/scratch/shared/beegfs/piyush/datasets/SSv2/20bn-something-something-v2/{}.webm",
    "cia-epic": "/scratch/shared/beegfs/piyush/datasets/EPIC-Kitchens-100/cut_clips/{}.MP4",
    "cia-charades": "/scratch/shared/beegfs/piyush/datasets/Charades/Charades_v1_480_cut_clips/{}.mp4",
    "neg-coco": "/scratch/shared/beegfs/piyush/datasets/COCO2017/val2017/{}.jpg",
    "neg-msrvtt": "/scratch/shared/beegfs/piyush/datasets/MSRVTT/videos/all/{}.mp4",
    "covr-webvid": "/datasets/WebVid/videos/{}.mp4"
}


def build_video_text_payload(video_path: str, edit_instruction: str, max_frames: int) -> dict:
    """Build a single entry for ``Qwen3VLEmbedder.process`` for composed video--text retrieval.

    ``video_edit_eol_prompt`` places ``<video>`` between the lead-in and the edit instruction;
    ``Qwen3VLEmbedder.format_model_input`` orders content as all videos, then all text parts, so
    we split on ``<video>`` into two text segments: [video][text_before][text_after].
    """
    templ = video_edit_eol_prompt().replace('<sent>', edit_instruction)
    if '<video>' not in templ:
        raise ValueError("video_edit_eol_prompt must contain a <video> placeholder")
    before, after = templ.split('<video>', 1)
    return {
        'video': video_path,
        'text': [before, after],
        'max_frames': max_frames,
    }


def compute_embedding(model, row: dict, path_map: dict, max_frames: int) -> torch.Tensor:
    """Encode one CSV row with ``Qwen3VLEmbedder``; return float32 CPU vector (unnormalized)."""
    modality = row['modality']
    source = row['source']

    if modality in ('text', 'text-standard', 'text-negation'):
        payload = {'text': row['value']}
    elif modality == 'image':
        payload = {'image': path_map[source].format(row['value'])}
    elif modality == 'video':
        payload = {
            'video': path_map[source].format(row['value']),
            'max_frames': max_frames,
        }
    elif modality == 'video-text':
        spec = eval(row['value'])
        video_path = path_map[source].format(spec['video'])
        payload = build_video_text_payload(video_path, spec['text'], max_frames)
    else:
        raise ValueError(f"Invalid modality: {modality}")

    emb = model.process([payload], normalize=False)
    return emb.squeeze(0).cpu().float()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path',
        type=str,
        default='/work/piyush/experiments/CaRe/Qwen3-VL-Embedding-8B/final-10112025/nli_9000+ego_1000+subj_replaced-seed_42',
    )
    parser.add_argument('--model_name', type=str, default='qwen3vlemb')
    parser.add_argument('--csv_path', type=str, default='./data/nuanced_retrieval_data-v1.csv')
    parser.add_argument(
        '--max_frames',
        type=int,
        default=16,
        help='Video sampling cap passed to Qwen3VLEmbedder for video / video-text rows.',
    )
    args = parser.parse_args()

    csv_path = args.csv_path
    csv_name = os.path.basename(csv_path).split('.')[0]
    assert os.path.exists(csv_path), f"CSV file does not exist: {csv_path}"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    df = df.dropna()
    print(f"Number of rows after removing null values: {len(df)}")

    from models.qwen3vl_embedding import Qwen3VLEmbedder

    model = Qwen3VLEmbedder(
        model_name_or_path=args.model_path,
        device_map='cuda',
        torch_dtype=torch.float16,
        attn_implementation='flash_attention_2',
    )
    su.misc.num_params(model.model)

    save_dir = f"{args.model_path}/embs"
    os.makedirs(save_dir, exist_ok=True)
    save_name = f"{args.model_name}_{csv_name}_embeddings.pt"
    save_path = os.path.join(save_dir, save_name)

    if os.path.exists(save_path):
        print(f"Embeddings already exist at {save_path}")
        embeddings = torch.load(save_path)
        df = df[~df['id'].isin(list(embeddings.keys()))]
        print(f"Number of rows to compute: {len(df)}")
    else:
        embeddings = {}

    for i in su.log.tqdm_iterator(range(len(df)), desc='Computing embeddings'):
        row = df.iloc[i].to_dict()
        try:
            z = compute_embedding(model, row, data_dir, max_frames=args.max_frames)
            z = torch.nn.functional.normalize(z, dim=-1)
            embeddings[row['id']] = z
        except Exception as exc:
            print(f"Error computing embedding for {row['id']}: {exc}")
            continue

    torch.save(embeddings, save_path)
    print(f"Saved embeddings to {save_path}")
    print(f"Number of embeddings: {len(embeddings)}")
