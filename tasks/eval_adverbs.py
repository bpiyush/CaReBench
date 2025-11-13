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

# model_id = "/work/piyush/experiments/CaRe/Tarsier-7b/nli-9k+ego4d-1k/merged_checkpoint"
model_id = "/work/piyush/experiments/CaRe/Tarsier-7b/final-10112025/"\
    "nli_9000+ego_1000+subj_replaced-seed_42/merged_checkpoint"
encoder = AutoEncoder.from_pretrained(model_id, device_map='cuda:0')
su.misc.num_params(encoder.model)


# video_dir = "/datasets/ActivityNet/2020-version/activitynet_frames"
# video_dir = "/scratch/shared/beegfs/piyush/datasets/ActivityNetCaptions/videos/v1-2/test/"
video_dir = "/scratch/shared/beegfs/piyush/datasets/MMEB-V2/video-tasks/frames/data/ziyan/video_retrieval/VATEX/frames"
data_dir = "/scratch/shared/beegfs/piyush/datasets/PseudoAdverbs"

df_anno = pd.read_csv(f"{data_dir}/datasets/VATEX_Adverbs/annotations.csv")
df_advb = pd.read_csv(f"{data_dir}/datasets/VATEX_Adverbs/adverbs.csv")
print(df_anno.shape, df_advb.shape)

df_anno['clip_dir'] = df_anno.clip_id.apply(lambda x: f"{video_dir}/{x}")
df_anno = df_anno[df_anno['clip_dir'].apply(os.path.isdir)]

print("Number of rows with clip directory available: ", len(df_anno))
# ext = 'mp4'
# df_anno['video_path'] = df_anno.clip_id.apply(lambda x: f"{video_dir}/{x}.{ext}")
# df_anno = df_anno[df_anno['video_path'].apply(os.path.isdir)]
# print("Number of rows with video available: ", len(df_anno))

adverb_to_antonym = dict(df_advb.values)
len(adverb_to_antonym)


def read_frames(clip_id, n_frames=8):
    frame_dir = f"{video_dir}/{clip_id}"
    paths = natsorted(glob(f"{frame_dir}/*"))
    sf = 0
    ef = len(paths)
    n_frames = min(n_frames, ef - sf)
    indices = np.linspace(sf, ef, n_frames, endpoint=False, dtype=int)
    paths = np.array(paths)[indices]
    frames = [PIL.Image.open(f).convert("RGB") for f in paths]
    x = torch.stack(
        [torch.from_numpy(np.asarray(f)) for f in frames]
    )
    x = x.permute(0, 3, 1, 2)  # (T, C, H, W), torch.uint8
    return x


def embed_video_action(clip_id, action, n_frames=16, verbose=False):

    PROMPT = "Video: <video>\n"\
             "Action: This video shows the action: <sent>\n"\
             "Look at the video carefully.\n"\
             "Summarize the action in the video in one word:"
    PROMPT = f"USER: {PROMPT} ASSISTANT: "

    generate_kwargs = {
        "max_new_tokens": 1,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
    }

    # Prepare video
    # pixel_values = read_frames_decord(video_path, n_frames)
    pixel_values = read_frames(clip_id, n_frames).unsqueeze(0)
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
        input_prompt = input_prompt.replace('<sent>', action)

        if verbose:
            print(input_prompt)
            print("-" * 120)

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

import ipdb; ipdb.set_trace()
i = 0
row = df_anno.iloc[i].to_dict()
zv = embed_video_action(
    row['clip_id'], row['clustered_action'], n_frames=8, verbose=True
)
zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float()
print(zv.shape)

def embed_adverb_action(adverb, action, n_frames=16, verbose=False):
    prompt = f"The action {action} is performed {adverb}."
    with torch.no_grad():
        zt = encoder.encode_text(prompt).cpu().squeeze(0).float()
        zt = torch.nn.functional.normalize(zt, dim=-1)
    return zt


text_embeds = {}
for i in su.log.tqdm_iterator(
    range(len(df_anno)), desc='Computing features for text'
):
    row = df_anno.iloc[i].to_dict()
    key = f"{row['clustered_action']}/{row['clustered_adverb']}"
    if key in text_embeds:
        continue
    zv = embed_adverb_action(row['clustered_adverb'], row['clustered_action'])
    zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float()
    text_embeds[key] = zv

    antonym = adverb_to_antonym[row['clustered_adverb']]
    key = f"{row['clustered_action']}/{antonym}"
    if key in text_embeds:
        continue
    zv = embed_adverb_action(antonym, row['clustered_action'])
    zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float()
    text_embeds[key] = zv


# Compute video-action embeddings
video_embeds = {}
n_frames = 8
for i in su.log.tqdm_iterator(
    range(len(df_anno)), desc='Computing features for videos'
):
    row = df_anno.iloc[i].to_dict()
    try:
        zv = embed_video_action(row['clip_id'], row['clustered_action'], n_frames=n_frames)
        zv = torch.nn.functional.normalize(zv, dim=-1).cpu().float()
        video_embeds[row['clip_id']] = zv
    except:
        print(f"Failed {row['clip_id']}.")
        continue
len(video_embeds)


correct = []
for i in range(len(df_anno)):
    row = df_anno.iloc[i].to_dict()
    clip_id = row['clip_id']
    action = row['clustered_action']
    adverb = row['clustered_adverb']
    try:
        zt_adverb = text_embeds[f"{action}/{adverb}"]
        zt_antonm = text_embeds[f"{action}/{adverb_to_antonym[adverb]}"]
        zv = video_embeds[clip_id]
        c = (zv @ zt_adverb) > (zv @ zt_antonm)
        correct.append(int(c))
    except:
        print(i)
        continue
accu = np.round(100. * np.mean(correct), 3)
print("Accuracy: ", accu)