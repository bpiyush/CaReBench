import torch
from models.qwen3vl_embedding import Qwen3VLEmbedder
import shared.utils as su

model_name_or_path = "/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B"
print(f"Loading model from {model_name_or_path}")
model = Qwen3VLEmbedder(
    model_name_or_path=model_name_or_path,
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",
    device_map="cuda:0",
)
su.misc.num_params(model.model)

queries = [
    {'text': "Someone is folding a paper."},
    {'text': "Someone is unfolding a paper."},
    {'text': "Someone is cutting a paper."},
]

frame_dir = '/users/piyush/projects/TimeBound.v1/sample_data'
# documents = [
#     {'video': [f"{frame_dir}/folding_paper-0.png", f"{frame_dir}/folding_paper-10.png", f"{frame_dir}/folding_paper-20.png", f"{frame_dir}/folding_paper-34.png"]},
# ]
documents = [
    {
        "video": f"{frame_dir}/folding_paper.mp4",
        "fps": 4,
        "max_frames": 16,
    }
]

inputs = queries + documents
embeddings = model.process(inputs)
print(f"Embeddings shape: {embeddings.shape}")
sim = embeddings[:len(queries)] @ embeddings[len(queries):].T
print(f"Similarity matrix shape: {sim.shape}")
print(f"Similarity matrix: {sim}")
