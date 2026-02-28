import os
import torch
import argparse

import shared.utils as su
from tasks.eval_chiral_retrieval import load_data
from utils.chiral_retrieval_metrics import (
    compute_metrics,
    print_metrics_as_latex_row,
)
import pandas as pd
import numpy as np
import json
from datasets import load_dataset


def load_data_video_cls(
    data_root='/scratch/shared/beegfs/piyush/datasets/MMEB-V2',
    cfg_path='/users/piyush/projects/VLM2Vec/experiments/public/eval/video_cls.yaml'
):
    # Load meta config
    meta_config = su.io.load_yml(cfg_path)

    # Generate dataframe of video paths
    df_video = []
    for ds_key in su.log.tqdm_iterator(meta_config, desc='Gathering video paths'):
        file_name = meta_config[ds_key]['json_name']
        data_file = f'{data_root}/video-tasks/data/{file_name}'
        assert os.path.exists(data_file)
        data = su.io.load_jsonl(data_file)

        ds_name = os.path.basename(meta_config[ds_key]['frame_root'])
        for d in data:
            video_id = d['video_id']
            video_dir = f"{data_root}/video-tasks/frames/{ds_name}/{video_id}"
            assert os.path.isdir(video_dir)
            df_video.append(
                dict(ds_key=ds_key, ds_name=ds_name, video_id=video_id, video_dir=video_dir)
            )
    df_video = pd.DataFrame(df_video)
    assert len(df_video.video_id.unique()) == len(df_video)
    return df_video


def load_data_video_ret(
    data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2',
    cfg_path = '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml',
    video_root = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames/data/ziyan/video_retrieval"
):
    # Load meta config
    meta_config = su.io.load_yml(cfg_path)

    # Load video root
    # This defines the huggingface repo and subset for each dataset
    # (repo, subset, split)
    json_paths = {
        "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
        "MSVD": ("VLM2Vec/MSVD", None, "test"),
        "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
        # "YouCook2": ("VLM2Vec/YouCook2", None, "val"), # HF version compatibility issue
        "YouCook2": ("lmms-lab/YouCook2", None, "val"),
        "VATEX": ("VLM2Vec/VATEX", None, "test"),
    }

    video_id_extractor = {
        "MSR-VTT": lambda x: x['video_id'],
        "MSVD": lambda x: x['video_id'],
        "DiDeMo": lambda x: x['video'].split('/')[-1].split('.')[0],
        "YouCook2": lambda x: x['id'],
        "VATEX": lambda x: x['videoID'],
    }


    df_video = {'video_id': [], 'video_dir': []}
    from datasets import load_dataset
    for ds_key in su.log.tqdm_iterator(meta_config, desc='Processing datasets'):
        print(ds_key)
        repo, subset, split = json_paths[ds_key]
        df = pd.DataFrame(load_dataset(repo, subset)[split])
        video_dir = f"{video_root}/{ds_key}/frames"
        video_ids = os.listdir(video_dir)
        assert len(video_ids) == len(df)
        # print(df.iloc[0])
        
        df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)
        df['video_dir'] = df['video_id'].apply(lambda x: f"{video_root}/{ds_key}/frames/{x}")
        df_video['video_id'].extend(df['video_id'].tolist())
        df_video['video_dir'].extend(df['video_dir'].tolist())
        print('-' * 100)
    df_video = pd.DataFrame(df_video)
    assert len(df_video.video_id.unique()) == len(df_video)
    return df_video


data_root = '/scratch/shared/beegfs/piyush/datasets/MMEB-V2'
meta_config = su.io.load_yml(
    '/users/piyush/projects/VLM2Vec/experiments/public/eval/video_ret.yaml'
)

# This defines the huggingface repo and subset for each dataset
# (repo, subset, split)
json_paths = {
    "MSR-VTT": ("VLM2Vec/MSR-VTT", "test_1k", "test"),
    "MSVD": ("VLM2Vec/MSVD", None, "test"),
    "DiDeMo": ("VLM2Vec/DiDeMo", None, "test"),
    "YouCook2": ("lmms-lab/YouCook2", None, "val"),
    "VATEX": ("VLM2Vec/VATEX", None, "test"),
}
video_id_extractor = {
    "MSR-VTT": lambda x: x['video_id'],
    "MSVD": lambda x: x['video_id'],
    "DiDeMo": lambda x: x['video'].split('/')[-1].split('.')[0],
    "YouCook2": lambda x: x["id"],
    "VATEX": lambda x: x['videoID'],
}
video_root = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames/data/ziyan/video_retrieval"
captions_extractor = {
    "MSR-VTT": lambda x: [x["caption"]],
    "MSVD": lambda x: x["caption"],
    "DiDeMo": lambda x: [x["caption"]],
    "YouCook2": lambda x: [x['sentence']],
    "VATEX": lambda x: x["enCap"],
}


def gather_text_embeddings(encoder, texts, ds_name):
    ZT = {}
    for text in su.log.tqdm_iterator(texts, desc=f'Computing text embeddings for {ds_name}'):
        with torch.no_grad():
            zt = encoder.encode_text(text)
            zt = torch.nn.functional.normalize(zt, dim=-1).squeeze(0).cpu().float()
        ZT[text] = zt
    return ZT


def compute_video_text_score(model, video_path, caption):
    import torch
    from models.tarsier2.dataset.utils import format_one_sample

    instruction = "Retrieval relevant text with user's query"
    prompt = (
        "Judge whether the Video meets the requirements based on the Query text "
        "and the Instruction provided. Note that the answer can only be 'yes' or 'no'.\n"
        f"Query: {caption}\n"
        f"Instruction: {instruction}"
    )

    sample = format_one_sample(media_file=video_path, prompt=prompt)
    sample = model.super_processor(sample)
    model_inputs = {
        k: v.to(model.model.device)
        for k, v in sample.items()
        if isinstance(v, torch.Tensor)
    }

    tokenizer = model.processor.tokenizer
    yes_token_id = tokenizer.encode("yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("no", add_special_tokens=False)[0]

    with torch.inference_mode():
        outputs = model.model.generate(
            **model_inputs,
            max_new_tokens=1,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=model.processor.tokenizer.eos_token_id
        )

    logits = outputs.scores[0][0]  # (vocab_size,)
    logit_yes = logits[yes_token_id].float()
    logit_no = logits[no_token_id].float()

    score = torch.sigmoid(logit_yes - logit_no).item()
    return score


if __name__ == "__main__":

    df_cls = load_data_video_cls()
    df_cls['video_path'] = df_cls['video_dir'].apply(lambda x: x.replace('video-tasks/frames', 'video-tasks/videos') + '.mp4')
    df_cls = df_cls[df_cls['video_path'].apply(os.path.exists)]

    df_ret = load_data_video_ret()
    df_ret['video_path'] = df_ret['video_dir'].apply(lambda x: x.replace('video-tasks/frames', 'video-tasks/videos') + '.mp4')
    df_ret = df_ret[df_ret['video_path'].apply(os.path.exists)]


    from models.modeling_encoders import EncoderForTarsier2

    # model_path = "/work/piyush/pretrained_checkpoints/Tarsier2-7b-0115"
    model_path = "/work/piyush/experiments/CaRe/Tarsier2-7b-0115/covr/chiral10k-covr10k/merged_checkpoint"
    model = EncoderForTarsier2.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
    )
    su.misc.num_params(model.model)


    # Load video embeddings
    feat_dir = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/features"

    video_embs_cls = torch.load(f"{feat_dir}/tarsier2-tara-cia10k-covr10k_video_embeddings_mmebv2_video_cls.pt")
    video_embs_ret = torch.load(f"{feat_dir}/tarsier2-tara-cia10k-covr10k_video_embeddings_mmebv2_video_ret.pt")

    # Only keep those rows with videos available
    df_cls = df_cls[df_cls.video_id.apply(lambda x: x in video_embs_cls)]
    df_ret = df_ret[df_ret.video_id.apply(lambda x: x in video_embs_ret)]

    ds_key = "MSR-VTT"
    su.log.print_update(f"Processing {ds_key}")
    d = meta_config[ds_key]

    repo, subset, split = json_paths[ds_key]
    df = pd.DataFrame(load_dataset(repo, subset)[split])
    
    video_dir = f"{video_root}/{ds_key}/frames"
    video_ids = os.listdir(video_dir)
    assert len(video_ids) == len(df)
    df['video_id'] = df.apply(lambda x: video_id_extractor[ds_key](x), axis=1)

    print(json.dumps(df.iloc[0].to_dict(), indent=2))

    all_texts = [
        captions_extractor[ds_key](df.iloc[i].to_dict()) for i in range(len(df))
    ]
    all_texts = np.unique(np.concatenate(all_texts))
    _text_embeds = gather_text_embeddings(model, all_texts, ds_key)
    
    zt = torch.stack([_text_embeds[t] for t in _text_embeds])
    zt = torch.nn.functional.normalize(zt, dim=-1)


    # Standard retrieval with reranking
    correct = []
    K = 20
    for i in su.log.tqdm_iterator(range(len(df)), desc='Reranking'):
        row = df.iloc[i].to_dict()
        video_id = row['video_id']
        video_path = df_ret[df_ret.video_id == video_id].iloc[0].video_path

        zv = video_embs_ret[video_id]
        zv = torch.nn.functional.normalize(zv, dim=-1)
        
        # Retrieve topK texts
        sims = (zv @ zt.T)
        topk = torch.topk(sims, k=K)
        indices = topk.indices

        # Re-ranking step
        scores = []
        captions = []
        for j in indices:
            cap = all_texts[j]
            s = compute_video_text_score(model, video_path, cap)
            scores.append(s)
            captions.append(cap)

        # Get prediction & gt
        pr = indices[np.argsort(-np.array(scores))[0]].item()
        gt = np.where(all_texts == row['caption'])[0][0]

        correct.append(gt == pr)
    correct = np.array(correct)
    import ipdb; ipdb.set_trace()