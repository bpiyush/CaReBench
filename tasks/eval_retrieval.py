import os
import sys
import json

import torch
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F
from utils.video import read_frames_decord
from torchvision.transforms.v2 import PILToTensor
from models.modeling_encoders import AutoEncoder

import shared.utils as su
from models.modeling_basemodels import EOL_PROMPTS
from qwen_vl_utils import process_vision_info


def get_video_embedding(duration, video_path, encoder):
    messages = [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                'fps': 32. / duration,
                "resized_height": 256,
                "resized_width": 455,
            },
            {"type": "text", "text": EOL_PROMPTS['video']},
        ],
    }]
    prompt = encoder.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    print(prompt)
    print(video_inputs[0].shape)

    inputs = encoder.processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        # **video_kwargs,
    )
    inputs = inputs.to("cuda")
    with torch.inference_mode():
        output = encoder.model.generate(
            **inputs,
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
        )
        z = output.hidden_states[0][-1][:, -1, :].cpu().float()
    return z


if __name__ == "__main__":
    num_frames = 32
    trim30 = False

    # Load model
    model_path = "/work/piyush/experiments/CaRe/special_milestones/care-stage2-nli-27k-ego4d-3k"
    encoder = AutoEncoder.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map='cuda:0',
    )
    
    # Load data
    data = "carebench"
    data_path = "./data.json"
    assert os.path.exists(data_path), f"{data_path} not found"
    with open(data_path) as f:
        data_configs = json.load(f)
    data_config = data_configs[data]
    anno_path = data_config['anno_path']
    with open(anno_path) as f:
        data = json.load(f)
        
    
    # Compute embeddings
    video_embs = []
    text_embs = []
    iterator = su.log.tqdm_iterator(data, desc='Computing embeddings')
    for item in iterator:
        video_path = f"{data_config['data_root']}/{item['video']}"
        assert os.path.exists(video_path)
        caption = item['caption']
        
        with torch.no_grad():
            zt = encoder.encode_text(caption).cpu().float()
        
        duration = su.video.get_duration(video_path)
        zv = get_video_embedding(duration, video_path, encoder)
        video_embs.append(zv)
        text_embs.append(zt)
        
    video_embs = torch.cat(video_embs)
    text_embs = torch.cat(text_embs)
    
    # Compute scores
    scores = text_embs @ video_embs.t()
    import ipdb; ipdb.set_trace()