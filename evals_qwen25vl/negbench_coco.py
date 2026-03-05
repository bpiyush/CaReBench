import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import einops

from shared.utils.log import tqdm_iterator


def recall_at_k(scores, positive_pairs, k):
    nb_texts, nb_images = scores.shape
    topk_indices = torch.topk(scores, k, dim=1)[1]
    nb_positive = positive_pairs.sum(dim=1)
    topk_indices_onehot = torch.nn.functional.one_hot(topk_indices, num_classes=nb_images)
    positive_pairs_reshaped = positive_pairs.view(nb_texts, 1, nb_images)
    nb_true_positive = (topk_indices_onehot * positive_pairs_reshaped).sum(dim=(1, 2))
    return nb_true_positive / nb_positive


def batchify(func, X, Y, batch_size, device, *args, **kwargs):
    results = []
    for start in range(0, len(X), batch_size):
        end = start + batch_size
        x = X[start:end].to(device)
        y = Y[start:end].to(device)
        result = func(x, y, *args, **kwargs).cpu()
        results.append(result)
    return torch.cat(results)


def embed_image(image_path):
    """Encode a single image using the model's image prompt template."""
    from copy import deepcopy
    from qwen_vl_utils import process_vision_info

    base_conversation = model.image_eol_prompt
    conversations = deepcopy(base_conversation)
    conversations[0]['content'] = [
        {"type": "image", "image": image_path},
        base_conversation[0]['content'][0],
    ]

    prompts = [
        model.processor.apply_chat_template(
            [conversations[0]], tokenize=False, add_generation_prompt=True
        )
    ]
    image_inputs, video_inputs = process_vision_info(conversations)

    inputs = model.processor(
        text=prompts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.model.device)
    with torch.inference_mode():
        output = model.model.generate(
            **inputs,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
    zi = output.hidden_states[0][-1][:, -1, :]
    return zi.squeeze(0).float()


def embed_text(text):
    """Encode a single text string using the model's text prompt template."""
    with torch.inference_mode():
        zt = model.encode_text(text).squeeze(0)
    return zt.float()


def gather_text_embeddings(df, index=0):
    texts_feat = {}
    failed = set()
    for i in tqdm_iterator(range(len(df)), desc=f'Computing text features (caption {index})'):
        text = eval(df.iloc[i].captions)[index]
        try:
            zt = embed_text(text)
            zt = torch.nn.functional.normalize(zt, dim=-1).cpu()
            texts_feat[i] = zt
        except Exception as e:
            print(f"Error computing text embedding for row {i}: {e}")
            failed.add(i)
    return texts_feat, failed


def gather_image_embs(df, image_dir):
    image_feats = {}
    failed = set()
    for i in tqdm_iterator(range(len(df)), desc='Computing image features'):
        row = df.iloc[i].to_dict()
        image_path = row['filepath'].replace('data/coco/images', image_dir)
        try:
            zi = embed_image(image_path)
            zi = torch.nn.functional.normalize(zi, dim=-1).cpu()
            image_feats[i] = zi
        except Exception as e:
            print(f"Error computing image embedding for row {i}: {e}")
            failed.add(i)
    return image_feats, failed


def compute_metrics(images_emb, texts_emb, text_to_image_index):
    scores = texts_emb @ images_emb.t()
    positive_pairs = torch.zeros_like(scores, dtype=bool)
    positive_pairs[torch.arange(len(scores)), text_to_image_index] = True

    metrics = {}
    for recall_k in [5]:
        metrics[f"image_retrieval_recall@{recall_k}"] = \
            (batchify(recall_at_k, scores, positive_pairs, 32, 'cpu', k=recall_k) > 0).float().mean().item()
        metrics[f"text_retrieval_recall@{recall_k}"] = \
            (batchify(recall_at_k, scores.T, positive_pairs.T, 32, 'cpu', k=recall_k) > 0).float().mean().item()
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", type=str,
        default="/work/piyush/pretrained_checkpoints/ArrowRL-Qwen2.5-VL-7B",
    )
    parser.add_argument("--model_name", type=str, default="qwen25vl")
    args = parser.parse_args()

    # Load model
    from models.qwen25vl import EncoderForQwen25VL
    model = EncoderForQwen25VL.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        attn_implementation='flash_attention_2',
        device_map='auto',
    )

    # Load NegBench COCO data
    data_dir = "/scratch/shared/beegfs/piyush/datasets/NegBench"
    image_dir = "/scratch/shared/beegfs/piyush/datasets/COCO2017"
    csv_name_std = "images/COCO_val_retrieval.csv"
    df_std = pd.read_csv(f"{data_dir}/{csv_name_std}")
    csv_name_neg = "images/COCO_val_negated_retrieval_llama3.1_rephrased_affneg_true.csv"
    df_neg = pd.read_csv(f"{data_dir}/{csv_name_neg}")

    # Gather text embeddings (5 captions per image)
    failed_indices = set()
    text_std_dicts = []
    text_neg_dicts = []
    for j in range(5):
        feat_std, fail_std = gather_text_embeddings(df_std, j)
        feat_neg, fail_neg = gather_text_embeddings(df_neg, j)
        text_std_dicts.append(feat_std)
        text_neg_dicts.append(feat_neg)
        failed_indices.update(fail_std)
        failed_indices.update(fail_neg)

    # Gather image embeddings
    image_feat_dict, fail_img = gather_image_embs(df_std, image_dir)
    failed_indices.update(fail_img)

    # Filter to rows that succeeded everywhere
    valid_indices = sorted(set(range(len(df_std))) - failed_indices)
    print(f"Valid rows: {len(valid_indices)} / {len(df_std)} (failed: {len(failed_indices)})")

    image_feat = torch.stack([image_feat_dict[i] for i in valid_indices])
    texts_feat_std_all = torch.stack([
        torch.stack([d[i] for i in valid_indices]) for d in text_std_dicts
    ])
    texts_feat_neg_all = torch.stack([
        torch.stack([d[i] for i in valid_indices]) for d in text_neg_dicts
    ])

    # Compute metrics
    text_std = einops.rearrange(texts_feat_std_all, 'j l d -> (j l) d')
    text_neg = einops.rearrange(texts_feat_neg_all, 'j l d -> (j l) d')
    text_to_image_index = np.arange(len(image_feat))
    text_to_image_index = np.concatenate([text_to_image_index] * 5)
    metrics = {
        'std': compute_metrics(image_feat, text_std, text_to_image_index),
        'neg': compute_metrics(image_feat, text_neg, text_to_image_index),
    }
    print(json.dumps(metrics, indent=4))

    # Save metrics
    result_dir = os.path.join(args.model_path, "metrics")
    os.makedirs(result_dir, exist_ok=True)
    metrics_path = os.path.join(result_dir, f"metrics_negbench_coco_{args.model_name}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")
