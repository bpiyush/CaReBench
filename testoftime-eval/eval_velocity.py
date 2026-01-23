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



from torch.utils.data import Dataset, DataLoader
from easydict import EasyDict as edict
import numpy as np
import json



class negDataset(Dataset):
    """
    Make this return all possibilities of captions.
    Dataset for video-to-text tasks.
    """

    def __init__(self, data_dict, frames_path, transform=None ):

        self.data_dict = data_dict
        self.frames_path = frames_path
        self.transform = transform


    def __getitem__(self, idx):

        test_name = self.data_dict[idx]['test_name']
        video_id, ev = self.data_dict[idx]['video_id'].split('.')[0], self.data_dict[idx]['event']
        pos_cap, neg_cap = self.data_dict[idx]['pos'], self.data_dict[idx]['neg']

        if self.transform:
            raise NotImplementedError("Frames tensor not implemented")
        else:
            frames = "{}/{}.mp4".format(self.frames_path, video_id)


        data = {
            "test_name": test_name,
            "video_id": video_id,
            "ev": ev,
            "frames": frames,
            "pos": pos_cap,
            "neg": neg_cap,
        }

        return data

    def __len__(self):
        return len(self.data_dict)


def get_data_local(dataset_path, frames_root, test_name=None, batch_size=1, num_workers=0, pin_memory=True, in_subset=True):
    """
    Load dataset directly from local path (git-cloned repository), bypassing HuggingFace cache.
    
    Args:
        dataset_path: Local path to the dataset directory (git-cloned VELOCITI repo).
                     Should contain test-data.parquet file.
    """
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    # Load dataset from parquet file in the git-cloned directory
    parquet_path = os.path.join(dataset_path, 'test-data.parquet')
    ds = load_dataset('parquet', data_files=parquet_path, cache_dir=None)
    # The parquet file loads as 'train' split by default, rename to 'test' for consistency
    if 'train' in ds:
        dataset = ds['train']
    else:
        dataset = ds[list(ds.keys())[0]]
    
    if in_subset:
        # Run only where item.is_subset is True
        dataset = dataset.filter(lambda example: example['in_subset'])

    if test_name:
        dataset = dataset.filter(lambda example: example['test_name'] == test_name)

    dataset = negDataset(dataset, frames_root, transform=None)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )

    return dataset, dataloader


def get_prompt(type='entailment'):

    entail_prompt = "Carefully watch the video and pay attention to the sequence of events, the details and actions of persons.\nHere is a caption that describes the video: {caption}\nBased on your observation, does the given video entail the caption? Just answer with either Yes or No. "

    mcq_prompt = "Carefully watch the video and pay attention to the sequence of events, the details and actions of persons. Here are two captions that describe the video. A) {cap1} B) {cap2}\nBased on your observation, select the caption that best describes the video. Just print either A or B."
    
    if type == 'entailment':
        return entail_prompt
    elif type == 'mcq':
        return mcq_prompt


def entail_score(logits, processor):
    import torch.nn.functional as F

    logits = F.softmax(logits, dim=-1)
    token_id_yes = processor.tokenizer.encode('Yes', add_special_tokens = False)[0]
    token_id_no  = processor.tokenizer.encode('No', add_special_tokens = False)[0]

    scores = logits[:,token_id_yes] / (logits[:,token_id_yes] + logits[:,token_id_no])
    return scores



def generate_logits(prompt, video_path, encoder, n_frames=16, verbose=False):
    generate_kwargs = {
        "max_new_tokens": 1,
        "output_hidden_states": True,
        "return_dict_in_generate": True,
        "output_scores": True,
    }

    # Prepare video
    pixel_values = read_frames_decord(video_path, n_frames)
    # pixel_values = read_frames(clip_id, n_frames).unsqueeze(0)
    pixel_values = transform_pixel_values(pixel_values)
    nframes = pixel_values.shape[1]
    to_image = ToPILImage()
    batched_frames = []
    for batch in pixel_values:
        frames = [to_image(v) for v in batch]
        batched_frames.append(frames)

    for frames in batched_frames:

        # Video
        input_prompt = prompt.replace("<video>", "<image>"*len(frames))

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
        # scores = processor.decode(outputs['sequences'][0][-1], skip_special_tokens=True)
        entailment_scores = entail_score(outputs['scores'][0], encoder.processor)
        return entailment_scores


def convert_to_prompt(messages):
    """
    Convert a list of message dictionaries to a prompt string.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' fields
        
    Returns:
        Formatted prompt string
    """
    prompt = ""
    
    for message in messages:
        role = message["role"].upper()
        prompt += f"{role}: "

        content_items = message["content"]
        for item in content_items:
            if item["type"] == "video":
                prompt += "<video>\n"
            elif item["type"] == "text":
                prompt += item["text"]
        
        prompt += " "
    
    prompt += "ASSISTANT: "
    
    return prompt


def eval_on_dataset(dataset, prompt, test_name, encoder):
    df = pd.DataFrame(
        columns=[
            "test_name",
            "video_id",
            "event",
            "pos_score",
            "neg_score",
        ]
    )
    cnt, tot = 0, 0
    from shared.utils.log import tqdm_iterator
    for item in tqdm_iterator(dataset, desc=f'Evaluating dataset on {test_name}', total=len(dataset)):
        prompt_pos = prompt.format(caption=item['pos'])
        prompt_neg = prompt.format(caption=item['neg'])
        video_path = item['frames']
        
        # Run on positive prompt
        messages = [
            
            {
                "role": "user",
                "content": [{"type": "video", "video": video_path, "fps": 8.0}, {"type": "text", "text": prompt_pos}]
            }
        ]
        input_prompt = convert_to_prompt(messages)
        pos_score = generate_logits(input_prompt, video_path, encoder, n_frames=16)
        
        # Run on negative prompt
        messages = [
            {
                "role": "user",
                "content": [{"type": "video", "video": video_path, "fps": 8.0}, {"type": "text", "text": prompt_neg}]
            }
        ]
        input_prompt = convert_to_prompt(messages)
        neg_score = generate_logits(input_prompt, video_path, encoder, n_frames=16)

        cnt += (pos_score > neg_score).sum().item()
        tot += 1

        tmp_df = pd.DataFrame(
                    {
                        "test_name": item["test_name"],
                        "video_id": item["video_id"],
                        "event": item["ev"],
                        "pos_score": pos_score.reshape(-1).detach().cpu().tolist(),
                        "neg_score": neg_score.reshape(-1).detach().cpu().tolist(),
                    }
                )
        df = pd.concat([df, tmp_df])
        df.to_csv('temp.csv', index=False)

    return (cnt / tot), df


if __name__ == "__main__":
    # Load model
    # model_id = "/work/piyush/pretrained_checkpoints/Tarsier-7b"
    model_id = "/work/piyush/pretrained_checkpoints/TARA"
    encoder = AutoEncoder.from_pretrained(model_id, device_map='cuda:0')
    su.misc.num_params(encoder.model)

    tests = {'action_adv', 'action_bind', 'action_manner', 'agent_bind', 'agent_random', 'chrono', 'control', 'coref'}
    dataset_path = "/scratch/shared/beegfs/piyush/datasets/VELOCITI"
    frames_root = "/scratch/shared/beegfs/piyush/datasets/VELOCITI/velociti_videos"
    batch_size = 1
    num_workers = 0
    pin_memory = True
    prompt = get_prompt()
    for test_name in tests:
        print('Evaluating', test_name)
        dataset, dataloader = get_data_local(
            dataset_path, 
            frames_root, 
            test_name=test_name,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )

        acc, df = eval_on_dataset(dataset, prompt, test_name, encoder)
        print("Test Name: ", test_name)
        print("Accuracy: ", acc)
        print("-" * 100)
        os.makedirs('./testoftime-eval/outputs/tara-entail', exist_ok=True)
        df.to_csv(f'./testoftime-eval/outputs/tara-entail/velocity_{test_name}.csv', index=False)
