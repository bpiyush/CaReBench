import os
import viclip
import torch
import warnings

warnings.filterwarnings("ignore")


class VideoProcessor:
    def __init__(self, n_frames=8, device=None):
        self.n_frames = n_frames
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
    
    def __call__(self, video_path):
        try:
            frames = viclip.load_video_frames(video_path, num_frames=self.n_frames)
            video_tensor = viclip.frames2tensor(frames, device=self.device)
            return video_tensor
        except Exception as e:
            print(f"Error loading video frames: {e}")
            video_tensor = torch.zeros(1, self.n_frames, 3, 224, 224).to(self.device)
            return video_tensor


class VideoFeatureComputer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def __call__(self, video_tensor):
        with torch.no_grad():
            return self.model.get_vid_features(video_tensor).cpu().float().squeeze(0)


class TextFeatureComputer:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, text_str):
        with torch.no_grad():
            return viclip.get_text_feat_dict([text_str], self.model, self.tokenizer, {})[text_str].cpu().float().squeeze(0)


def load_model_viclip(ckpt_path=None, size='b', n_frames=8):
    if ckpt_path is None:
        ckpt_path = os.environ.get('VICLIP_B_CKPT', '/work/piyush/pretrained_checkpoints/ViCLIP/ViCLIP-B_InternVid-FLT-10M.pth')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output = viclip.get_viclip(size=size, pretrain=ckpt_path)
    model, tokenizer = output['viclip'], output['tokenizer']
    model = model.to(device)
    
    vp = VideoProcessor(n_frames=n_frames, device=device)
    vfc = VideoFeatureComputer(model, device)
    tfc = TextFeatureComputer(model, tokenizer, device)
    
    return vp, vfc, tfc


if __name__ == "__main__":
    vp, vfc, tfc = load_model_viclip()
    file_path = '../TimeBound.v1/sample_data/folding_paper.mp4'
    video_tensor = vp(file_path)
    print(f"Video tensor shape: {video_tensor.shape}")
    zv = vfc(video_tensor)
    print(f"Video shape: {zv.shape}")

    zt = tfc('a photo of a cat')
    print(f"Text shape: {zt.shape}")