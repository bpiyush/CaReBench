import os
import av
import torch
import numpy as np

from transformers import AutoProcessor, AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download

np.random.seed(0)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`list[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`list[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices


class VideoProcessorOld:
    def __init__(self, n_frames=8):
        self.n_frames = n_frames
    
    def __call__(self, video_path):
        container = av.open(video_path)
        indices = sample_frame_indices(
            clip_len=self.n_frames,
            frame_sample_rate=1,
            seg_len=container.streams.video[0].frames,
        )
        return torch.from_numpy(read_video_pyav(container, indices))


from utils.video import read_frames_decord
class VideoProcessor:
    def __init__(self, n_frames=8):
        self.n_frames = n_frames
    
    def __call__(self, video_path):
        video = read_frames_decord(video_path, self.n_frames)
        return video


class VideoFeatureComputer:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
    
    def __call__(self, video_tensor):
        inputs = self.processor(videos=list(video_tensor), return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            return self.model.get_video_features(**inputs).squeeze(0).cpu().float()


class TextFeatureComputer:
    def __init__(self, model, processor, device):
        self.model = model
        self.processor = processor
        self.device = device
    
    def __call__(self, text_str):
        inputs = self.processor(text=text_str, padding=True, return_tensors="pt")
        inputs = inputs.to(self.device)
        with torch.no_grad():
            return self.model.get_text_features(**inputs).squeeze(0).cpu().float()


def load_model_xclip(model_id="microsoft/xclip-base-patch32"):
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    vp = VideoProcessor(n_frames=8)
    vfc = VideoFeatureComputer(model, processor, device)
    tfc = TextFeatureComputer(model, processor, device)

    return vp, vfc, tfc


if __name__ == "__main__":
    vp, vfc, tfc = load_model_xclip()
    file_path = '../TimeBound.v1/sample_data/folding_paper.mp4'
    video_tensor = vp(file_path)
    zv = vfc(video_tensor)
    print(f"Video shape: {zv.shape}")

    zt = tfc('a photo of a cat')
    print(f"Text shape: {zt.shape}")
