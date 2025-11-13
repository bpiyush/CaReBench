import os
import sys
import shutil
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import decord
import json
from IPython.display import display, Markdown, Latex
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"
import PIL.Image
from glob import glob
from natsort import natsorted

import shared.utils as su
from utils.video import read_frames_decord
from utils.model import transform_pixel_values
from torchvision.transforms.v2 import (
    ToPILImage,
)
from models.modeling_encoders import AutoEncoder


# Load model: global
model_id = "/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/"\
    "nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint"
encoder = AutoEncoder.from_pretrained(model_id, device_map='cuda:0')
su.misc.num_params(encoder.model)


data_dir = "/scratch/shared/beegfs/piyush/datasets/Ego4D-HCap/egocvr"
save_dir = f"{data_dir}/embs"
os.makedirs(save_dir, exist_ok=True)
clip_dir = f"{data_dir}/clips"
# fullvid_dir = "/scratch/shared/beegfs/shared-datasets/EGO4D/ego4d_data_v1/full_scale"
def load_data():
    df_anno = pd.read_csv(f"{data_dir}/egocvr_annotations.csv")
    df_base = pd.read_csv(f"{data_dir}/egocvr_data.csv")
    df_gall = pd.read_csv(f"{data_dir}/egocvr_annotations_gallery.csv")
    return df_anno, df_base, df_gall


def load_local_gallery_video(clip_name, n_frames=8, as_tensor=False):
    video_id, st, et = clip_name.split("_")
    # st = float(st.split('-')[0])
    # et = float(et.split('-')[0])
    # video_path = f"{fullvid_dir}/{video_id}.mp4"
    st = int(st.split('-')[0])
    et = int(et.split('-')[0])
    
    clip_name = "_".join([video_id, str(st), str(et)])
    video_path = f"{clip_dir}/{clip_name}.mp4"
    # frames = su.video.load_frames_linspace(
    #     video_path, st=st, et=et, n=n_frames, width=360, height=240,
    # )
    frames = su.video.load_frames_linspace(
        video_path, n=n_frames, width=360, height=240,
    )
    if as_tensor:
        frames = torch.from_numpy(np.stack([np.asarray(x) for x in frames]))
        frames = frames.permute((0, 3, 1, 2))
    return frames


PROMPT = "<video>\nEdit instruction: <sent>\n"\
         "Imagine the given text edit instruction applied on the given video.\n"\
         "Summarize the resulting video in one word:"
PROMPT = f"USER: {PROMPT} ASSISTANT: "
print(PROMPT)
def embed_video_text(video_path, edit_text, n_frames=8, verbose=False):
    generate_kwargs = {
        "max_new_tokens": 1,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }

    # Prepare video
    pixel_values = read_frames_decord(video_path, n_frames).unsqueeze(0)
    pixel_values = transform_pixel_values(pixel_values)
    nframes = pixel_values.shape[1]
    to_image = ToPILImage()
    batched_frames = []
    for batch in pixel_values:
        frames = [to_image(v) for v in batch]
        batched_frames.append(frames)

    for frames in batched_frames:

        # Video
        input_prompt = PROMPT.replace("<video>", "<image>"*len(frames))

        # Text
        input_prompt = input_prompt.replace('<sent>', edit_text)

        if verbose:
            print(input_prompt)

        input_ids = encoder.processor.get_text_inputs(input_prompt)
        frames = encoder.processor.get_pixel_values(frames)
        inputs = {
            "input_ids": input_ids,
            "pixel_values": frames
        }
        inputs = {k:v.to(encoder.model.device) for k,v in inputs.items() if v is not None}
        outputs = encoder.model.generate(
            **inputs,
            **generate_kwargs,
        )
        zv = outputs.hidden_states[0][-1][:, -1, :]
        break # Safe to break since it is just one video

    if verbose:
        print(zv.shape)

    return zv.squeeze(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--si", type=int, default=0)
    parser.add_argument("--ei", type=int, default=None)
    args = parser.parse_args()
    
    # Load data
    su.log.print_update('Loading data')
    df_anno, df_base, df_gall = load_data()
    print("Number of rows in df_anno: ", len(df_anno))
    print("Number of rows in df_base: ", len(df_base))
    print("Number of rows in df_gall: ", len(df_gall))
    su.log.print_update('')
    
    # [A] Compute candidate embeddings
    _use_candidate_embeddings = True
    si = args.si
    ei = args.ei if args.ei is not None else len(df_base)
    save_name = f"tarsier+tara_local_gallery_candidate_embeddings_egocvr-{si}-{ei}.pt"
    save_path = os.path.join(save_dir, save_name)
    if _use_candidate_embeddings:
        if os.path.exists(save_path):
            video_embeddings_local_clips = torch.load(save_path)
            print("Loaded candidate embeddings from ", save_path)
        else:

            su.log.print_update('Computing candidate embeddings')

            # 1. Collect all clip names in the gallery
            local_clip_names = []
            for i in range(len(df_gall)):
                row = df_gall.iloc[i].to_dict()
                # local candidates
                df_local = df_base.iloc[eval(row['local_idx'])]
                local_clip_names.extend(df_local.clip_name.tolist())
            local_clip_names = np.unique(np.array(local_clip_names))
            print("Total number of local clips in gallery: ", len(local_clip_names))
            
            # Filter local clip names
            local_clip_names = local_clip_names[si:ei]
            print(f"Running from {si} to {ei}.")

            # 2. Compute video embeddings for the local clips
            video_embeddings_local_clips = {}
            iterator = su.log.tqdm_iterator(local_clip_names, desc='Computing video features (local clips)')
            for clip_name in iterator:
                try:
                    video_tensor = load_local_gallery_video(clip_name, as_tensor=True)
                    with torch.no_grad():
                        zv = encoder.encode_vision(video_tensor.unsqueeze(0))
                        zv = torch.nn.functional.normalize(zv, dim=-1).cpu().squeeze(0).float()
                    video_embeddings_local_clips[clip_name] = zv
                except:
                    print(f"Error computing candidate embedding for {clip_name}. Skipping.")
                    continue
            torch.save(video_embeddings_local_clips, save_path)
    
    # [B] Compute query embeddings
    _use_query_embeddings = False
    save_name = f"tarsier+tara_query_embeddings_egocvr.pt"
    save_path = os.path.join(save_dir, save_name)
    if _use_query_embeddings:
        if os.path.exists(save_path):
            query_embeddings = torch.load(save_path)
            print("Loaded query embeddings from ", save_path)
        else:
            su.log.print_update('Computing query embeddings')
            query_embeddings = {}
            for i in su.log.tqdm_iterator(range(len(df_gall)), desc='Computing query embeddings'):
                row = df_gall.iloc[i].to_dict()
                query_video_path = f"{clip_dir}/{row['video_clip_id']}.mp4"
                edit_instruction = row['instruction']
                try:
                    zq = embed_video_text(query_video_path, edit_instruction)
                    zq = torch.nn.functional.normalize(zq, dim=-1)
                    zq = zq.cpu().float()
                    query_embeddings[f"{row['video_clip_id']}|{edit_instruction}"] = zq
                except:
                    print(f"Error computing query embedding for {i}. Skipping.")
                    continue
            torch.save(query_embeddings, save_path)