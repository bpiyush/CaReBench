import os
import sys
import json

import torch
import torch.nn.functional as F
import numpy as np
import einops
import matplotlib.pyplot as plt

# from utils.video import read_frames_decord
# from models.modeling_encoders import AutoEncoder
from models.qwen3vl_embedding import Qwen3VLEmbedder
import shared.utils as su
from notebooks.eval_care_retrieval import load_data
from notebooks.eval_care_retrieval import compute_retrieval_metrics_with_subsets


def gather_text_features(df):
    """Compute text embeddings for all unique text IDs, keyed by text_id."""
    text_ids = df['text_id'].unique()
    texts_feat = {}
    for text_id in su.log.tqdm_iterator(text_ids, desc='Computing text features'):
        text = df[df.text_id == text_id].template.unique()[0]
        emb = model.process([{'text': text}])
        zt = emb.squeeze(0).cpu().float()
        texts_feat[text_id] = zt
    return texts_feat


def gather_video_features(
    df,
    # fps,
    # max_frames,
):
    """Compute video embeddings for all unique video IDs."""
    video_ids = df['id'].unique()
    video_feat = {}
    for j, video_id in enumerate(
        su.log.tqdm_iterator(video_ids, desc='Computing video features')
    ):
        video_path = df[df['id'] == video_id].video_path.unique()[0]

        try:
            emb = model.process([{
                'video': video_path,
                # 'fps': fps,
                # 'max_frames': max_frames,
                'nframes': 16,
            }])
            zv = emb.squeeze(0).cpu().float()
            video_feat[video_id] = zv
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
    return video_feat


def encode_single_shot_query(
    model_wrapper,
    text_query: str,
    pos_video_path: str,
    neg_video_path: str,
    nframes: int = 8,
) -> torch.Tensor:
    """
    Encode a single-shot query with a positive video example,
    a negative video example, and a text query.
    Returns an L2-normalized embedding vector of shape (D,).
    """

    def _video_entry(path: str) -> dict:
        url = path if path.startswith(('http://', 'https://')) else 'file://' + path
        return {'type': 'video', 'video': url, 'nframes': nframes, 'total_pixels': model_wrapper.total_pixels}

    conversation = [
        {
            'role': 'system',
            'content': [{'type': 'text', 'text': model_wrapper.default_instruction}],
        },
        {
            'role': 'user',
            'content': [
                _video_entry(pos_video_path),
                {
                    'type': 'text',
                    'text': f'The above video is a positive example of "{text_query}".\n\n',
                },
                _video_entry(neg_video_path),
                {
                    'type': 'text',
                    'text': (
                        f'The above video is a negative example, i.e., NOT "{text_query}".\n\n'
                        f'{text_query}\n'
                        f'Summary above sentence in one word:'
                    ),
                },
            ],
        },
    ]

    processed_inputs = model_wrapper._preprocess_inputs([conversation])
    processed_inputs = {k: v.to(model_wrapper.model.device) for k, v in processed_inputs.items()}

    outputs = model_wrapper.forward(processed_inputs)
    emb = model_wrapper._pooling_last(outputs['last_hidden_state'], outputs['attention_mask'])
    emb = F.normalize(emb, p=2, dim=-1).squeeze(0).cpu().float()
    return emb


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B/')
    parser.add_argument('--model_name', type=str, default='qwen3vlemb')
    parser.add_argument("--device_map", type=str, default='auto')
    parser.add_argument('--dataset', type=str, default='ssv2')
    parser.add_argument('--split', type=str, default='validation')
    args = parser.parse_args()

    model = Qwen3VLEmbedder(
        model_name_or_path=args.model_path,
        device_map='cuda',
        torch_dtype=torch.float16,
        attn_implementation='flash_attention_2',
    )
    su.misc.num_params(model.model)

    df_train = load_data(dataset=args.dataset, split="train")
    df_train = df_train.drop_duplicates(subset=["video_path"])
    df_valid = load_data(dataset=args.dataset, split="validation")
    df_valid = df_valid.drop_duplicates(subset=["video_path"])
    
    # Compute candidate embeddings
    # Load video embeddings since they are already computed
    embs_dir = os.path.join(args.model_path, "embs")
    embs_path = os.path.join(embs_dir, f"video_feat-{args.dataset}.pt")
    if not os.path.exists(embs_path):
        raise ValueError(f"Video embeddings not found at {embs_path}")
    print(f"Loading video embeddings from {embs_path}")
    candidate_embeds = torch.load(embs_path, weights_only=False)
    print(f"Loaded {len(candidate_embeds)} video embeddings")



    # Usual query embeddings: query is a text
    query_embeds = gather_text_features(df_valid)
    
    
    # Construct queries composed of text and positive/negative video
    df = df_valid.copy()
    text_ids = df['text_id'].unique()
    queries = []
    for text_id in text_ids:
        text = df[df.text_id == text_id].template.unique()[0]

        # Find a positive video
        row_pos = df_train[df_train.text_id == text_id].sample(n=1).iloc[0].to_dict()

        # Find a (chiral) negative video
        text_id_neg = text_id.split("_")[0] + "_" + str(float(int(not int(float(text_id.split("_")[1])))))
        row_neg = df_train[df_train.text_id == text_id_neg].sample(n=1).iloc[0].to_dict()

        queries.append(
            {"text": text, "text_id": text_id, "video_pos": row_pos['video_path'], "video_neg": row_neg['video_path']}
        )

    single_shot_query_embeds = {}
    for i in su.log.tqdm_iterator(range(len(queries)), desc='Computing (single-shot) text features'):
        q = queries[i]
        single_shot_query_embeds[q['text_id']] = encode_single_shot_query(
            model, q['text'], q['video_pos'], q['video_neg']
        )
    
    
    metrics_pair_vanilla = compute_retrieval_metrics_with_subsets(
        df_valid, candidate_embeds, query_embeds, agg_col='chiral_triplet_id', verbose=False, video_id_col='id',
    )
    metrics_pair_single_shot = compute_retrieval_metrics_with_subsets(
        df_valid, candidate_embeds, single_shot_query_embeds, agg_col='chiral_triplet_id', verbose=False, video_id_col='id',
    )
    save_dict = {
        "metrics_pair_vanilla": metrics_pair_vanilla,
        "metrics_pair_single_shot": metrics_pair_single_shot,
    }
    save_dir = os.path.join(args.model_path, "metrics")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"metrics_single_shot_t2v_{args.model_name}_{args.dataset}_{args.split}.json")
    with open(save_path, "w") as f:
        json.dump(save_dict, f)
    print("Saved metrics to: ", save_path)
    print("Basic text query: ", metrics_pair_vanilla['txt_r1'])
    print("Single-shot text query: ", metrics_pair_single_shot['txt_r1'])