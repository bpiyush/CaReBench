import sys
model_path = "/work/piyush/pretrained_checkpoints/Qwen3-VL-Embedding-8B"
sys.path.insert(0, model_path)
import torch

from local_scripts.qwen3_vl_embedding import Qwen3VLEmbedder

# Define a list of query texts
queries = [
    {"text": "A woman playing with her dog on a beach at sunset."},
    {"text": "Pet owner training dog outdoors near water."},
    {"text": "Woman surfing on waves during a sunny day."},
    {"text": "City skyline view from a high-rise building at night."},
    {"text": "A woman folding a paper in a park."}
]

# Define a list of document texts and images
documents = [
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
    {"image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"text": "A woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust.", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
    {"video": "/users/piyush/projects/TimeBound.v1/sample_data/folding_paper.mp4"}
]


# Initialize the Qwen3VLEmbedder model
model = Qwen3VLEmbedder(model_name_or_path=model_path, device_map="cuda", attn_implementation="flash_attention_2", dtype=torch.float16)
# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# model = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, torch_dtype=torch.float16, attn_implementation="flash_attention_2")

# Combine queries and documents into a single input list
inputs = queries + documents

# Process the inputs to get embeddings
embeddings = model.process(inputs)

# Compute similarity scores between query embeddings and document embeddings
n_queries = len(queries)
similarity_scores = (embeddings[:n_queries] @ embeddings[n_queries:].T)

# Print out the similarity scores in a list format
print(similarity_scores)

# [[0.74267578125, 0.6630859375, 0.6328125], [0.443603515625, 0.33349609375, 0.396484375], [0.3671875, 0.2354736328125, 0.289306640625], [0.060821533203125, -0.01557159423828125, 0.0165863037109375]]
