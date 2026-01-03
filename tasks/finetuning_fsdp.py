import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Get environment variables (set by torchrun)
rank = int(os.environ.get("RANK", 0))
local_rank = int(os.environ.get("LOCAL_RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Initialize process group
if not dist.is_initialized():
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )

# Set device for this rank
device = torch.device(f"cuda:{local_rank}")
torch.cuda.set_device(device)

if rank == 0:
    print(f"Initialized distributed training with {world_size} GPUs")
    print("Strategy: All ranks load to CPU, FSDP shards to GPUs (~8.5GB each)")

# Model path
model_name_or_path = "/work/piyush/pretrained_checkpoints/Tarsier-34b"
llm_path = model_name_or_path + "-llm"

if not os.path.exists(llm_path):
    raise RuntimeError(
        f"LLM weights not found at {llm_path}. "
        f"Run `python tasks/split_weights.py -m {model_name_or_path}` first."
    )

# Step 1: Auto Wrap Policy
auto_wrap_policy = partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={LlamaDecoderLayer}
)

# Step 2: Mixed Precision
mixed_precision_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.bfloat16,
    buffer_dtype=torch.bfloat16,
)

# No CPU offload - sharded params stay on GPU (plenty of room: 8.5GB per GPU)

# Step 4: Load model on ALL ranks to CPU
# Each rank loads independently - this works because low_cpu_mem_usage=True
# loads incrementally and we have 376GB total RAM
if rank == 0:
    print(f"All ranks loading model from {llm_path} to CPU...")
    print("Using low_cpu_mem_usage=True for incremental loading...")

# Stagger loading to avoid simultaneous disk I/O
dist.barrier(device_ids=[local_rank])
for r in range(world_size):
    if rank == r:
        print(f"Rank {rank}: Loading model to CPU...")
        model = LlamaForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        print(f"Rank {rank}: Model loaded to CPU")
    dist.barrier(device_ids=[local_rank])

if rank == 0:
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {total_params / 1e9:.2f}B parameters")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(llm_path)

dist.barrier(device_ids=[local_rank])

if rank == 0:
    print("\nWrapping with FSDP...")
    print("Each GPU will hold 1/8 of params (~8.5GB) permanently on GPU")

# Step 5: Wrap with FSDP
# All ranks have same model on CPU
# FSDP will shard and move only 1/N of params to each GPU
# No sync needed since all ranks loaded same weights
model = FSDP(
    model,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mixed_precision_policy,
    device_id=device,
    sync_module_states=False,  # All ranks have same weights, no broadcast needed
    limit_all_gathers=True,
)

if rank == 0:
    print("FSDP wrapping complete!")

dist.barrier(device_ids=[local_rank])

# Verify memory
torch.cuda.synchronize()
gpu_allocated = torch.cuda.memory_allocated(device) / 1024**3
gpu_reserved = torch.cuda.memory_reserved(device) / 1024**3

print(f"Rank {rank} (GPU {local_rank}): {gpu_allocated:.2f}GB allocated, {gpu_reserved:.2f}GB reserved")

dist.barrier(device_ids=[local_rank])

if rank == 0:
    print("\n✓ Model wrapped with FSDP!")
    print("Sharded params are now on GPUs. Ready for training!")

# Cleanup
if dist.is_initialized():
    dist.destroy_process_group()
