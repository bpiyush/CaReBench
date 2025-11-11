import torch
import torch.nn.functional as F
import requests
from PIL import Image
from transformers import AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from utils.video import read_frames_decord


llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'
img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
vid_prompt = llama3_template.format('<video>\nSummary above video in one word: ')


class VideoProcessor:
    def __init__(self, n_frames=16):
        self.n_frames = n_frames
    
    def __call__(self, video_path):
        video = read_frames_decord(video_path, self.n_frames)
        return video.unsqueeze(0)


class VideoFeatureComputer:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def __call__(self, video_tensor):
        assert video_tensor.ndim == 5, "Video tensor must be batch of videos."
        prompts = [vid_prompt.replace(f"<video>\n", f"<image>\n" * len(p)) for p in video_tensor]
        inputs = self.processor(text=prompts, images=list(video_tensor[0]), padding=True, return_tensors="pt").to('cuda')
        with torch.no_grad():
            output = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            output = F.normalize(output, dim=-1).cpu().float().squeeze(0)
        return output


class TextFeatureComputer:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def __call__(self, text):
        prompt = text_prompt.replace('<sent>', text)
        inputs = self.processor(text=[prompt], return_tensors="pt", padding=True).to('cuda')
        with torch.no_grad():
            output = self.model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
            output = F.normalize(output, dim=-1).cpu().float().squeeze(0)
        return output


def load_model_e5v():
    model_path = '/work/piyush/pretrained_checkpoints/e5-v'
    processor = LlavaNextProcessor.from_pretrained(model_path)
    model = LlavaNextForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
    
    vp = VideoProcessor(n_frames=16)
    vfc = VideoFeatureComputer(model, processor)
    tfc = TextFeatureComputer(model, processor)
    return vp, vfc, tfc
    

if __name__ == "__main__":
    vp, vfc, tfc = load_model_e5v()
    text = 'A dog sitting in the grass.'
    text_emb = tfc(text)
    print(text_emb.shape)
    
    video_path = '../TimeBound.v1/sample_data/folding_paper.mp4'
    video_tensor = vp(video_path)
    video_emb = vfc(video_tensor)
    print(video_emb.shape)


# img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
# text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')
# vid_prompt = llama3_template.format('<video>\nSummary above video in one word: ')

# image_paths = ['../TimeBound.v1/sample_data/folding_paper.png']
# images = [Image.open(f) for f in image_paths]

# texts = ['A dog sitting in the grass.',
#          'A cat standing in the snow.']

# text_inputs = processor([text_prompt.replace('<sent>', text) for text in texts], return_tensors="pt", padding=True).to('cuda')
# img_inputs = processor([img_prompt]*len(images), images, return_tensors="pt", padding=True).to('cuda')

# with torch.no_grad():
#     text_embs = model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
#     img_embs = model(**img_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]

#     text_embs = F.normalize(text_embs, dim=-1)
#     img_embs = F.normalize(img_embs, dim=-1)

# print(text_embs @ img_embs.t())

# video_path = '../TimeBound.v1/sample_data/folding_paper.mp4'
# video_tensor = read_frames_decord(video_path, 16).unsqueeze(0)

# prompts = [vid_prompt.replace(f"<video>\n", f"<image>\n" * len(p)) for p in video_tensor]
# inputs = processor(text=prompts, images=list(video_tensor[0]), padding=True, return_tensors="pt").to('cuda')
# with torch.no_grad():
#     output = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
#     output = F.normalize(output, dim=-1).cpu().float().squeeze(0)
# print(output.shape)
# import ipdb; ipdb.set_trace()