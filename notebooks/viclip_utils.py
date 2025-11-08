import os
os.environ['VICLIP_B_CKPT'] = '/work/piyush/pretrained_checkpoints/ViCLIP/ViCLIP-B_InternVid-FLT-10M.pth'

import viclip
import torch
import warnings

warnings.filterwarnings("ignore")

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
output = viclip.get_viclip(size='b', pretrain=os.environ['VICLIP_B_CKPT'])
model, tokenizer = output['viclip'], output['tokenizer']

# Move to device
model = model.to(device)

# Load video frames
video_path = "../TimeBound.v1/sample_data/folding_paper.mp4"
frames = viclip.load_video_frames(video_path, num_frames=8)
video_tensor = viclip.frames2tensor(frames, device=device)
zv = model.get_vid_features(video_tensor).cpu().float().squeeze(0)
print(f"Video shape: {zv.shape}")

# Text features
text = "a person folding paper"
# zt = model.get_text_features(text)
zt = viclip.get_text_feat_dict([text], model, tokenizer, {})[text].cpu().float().squeeze(0)
print(f"Text shape: {zt.shape}")

# # Define text candidates
# texts = [
#     "a person folding paper",
#     "a cat playing with a ball",
#     "a dog running in the park",
#     "a bird flying in the sky",
#     "water flowing in a river"
# ]

# # Perform retrieval
# retrieved_texts, probabilities = viclip.retrieve_text(
#     frames, 
#     texts, 
#     models={'viclip': model, 'tokenizer': tokenizer},
#     topk=3,
#     device=device
# )

# print("Top retrieved texts:")
# for text, prob in zip(retrieved_texts, probabilities):
#     print(f"  {text}: {prob:.3f}")


# # Compute video embedding for this video
# zv = viclip.encode_vision(frames, model=model, device=device)