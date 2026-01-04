"""
FSDP Fine-tuning for Large Language Models (34B+)

This script uses PyTorch FSDP to shard a large model across multiple GPUs
for memory-efficient training.

Usage:
    torchrun --nproc_per_node=8 tasks/finetuning_fsdp.py \
        --model_name_or_path /path/to/model \
        --data_path /path/to/data.csv \
        --output_dir ./output
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Fix for OOM error: Set expandable_segments to True to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import fire
import torch
import torch.distributed as dist
import transformers
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, LlamaForCausalLM, set_seed
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# Import from existing finetuning.py
from tasks.finetuning import (
    DataCollatorForSeq2SeqForNeg,
    SentembTrainer,
    generate_sentemb_prompt,
)
from utils.model import EOL_PROMPTS


class FSDPCompatibleSentembTrainer(SentembTrainer):
    """SentembTrainer that works with manually wrapped FSDP models."""
    
    def _wrap_model(self, model, training=True):
        """Override to prevent Trainer from wrapping an already FSDP-wrapped model."""
        # If model is already wrapped with FSDP, don't wrap again
        if isinstance(model, FSDP):
            return model
        # Otherwise, use parent's method
        return super()._wrap_model(model, training)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Get device from model (FSDP-aware)
        device = next(model.parameters()).device
        
        if self.is_nli and self.use_neg_sentence:
            input_ids, labels, neg = inputs["input_ids"], inputs["labels"], inputs['attention_mask']
            pad_token_id = self.tokenizer.pad_token_id
            if self.fix_attention_mask:
                labels[labels < 0 ] = pad_token_id
                neg[neg < 0] = pad_token_id
            else:
                labels[labels < 0 ] = 0
                neg[neg < 0] = 0
            # padding tensor length
            mw = max(input_ids.size(1), labels.size(1), neg.size(1))

            pad_size = mw - labels.size(1)
            if pad_size > 0:
                label_pads = torch.zeros(labels.size(0), pad_size, device=device).long()
                label_pads.fill_(pad_token_id)
                labels = torch.cat([label_pads, labels], dim=1)
            pad_size = mw - input_ids.size(1)
            if pad_size > 0:
                input_pads = torch.zeros(input_ids.size(0), pad_size, device=device).long()
                input_pads.fill_(pad_token_id)
                input_ids = torch.cat([input_pads, input_ids], dim=1)
            pad_size = mw - neg.size(1)
            if pad_size > 0:
                neg_pads = torch.zeros(neg.size(0), pad_size, device=device).long()
                neg_pads.fill_(pad_token_id)
                neg = torch.cat([neg_pads, neg], dim=1)

            inputs['input_ids'] = torch.cat([input_ids, labels, neg], dim=0)
            if self.fix_attention_mask:
                inputs['attention_mask'] = (inputs['input_ids'] != pad_token_id).long()
            else:
                inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
            del inputs['labels']
        elif self.is_nli:
            input_ids, labels = inputs["input_ids"], inputs["labels"]
            labels[labels < 0 ] = 0
            # padding tensor length
            if input_ids.size(1) > labels.size(1):
                pad_size = input_ids.size(1) - labels.size(1)
                labels = torch.cat([torch.zeros(labels.size(0), pad_size, device=device).long(), labels], dim=1)
            else:
                pad_size = labels.size(1) - input_ids.size(1)
                input_ids = torch.cat([torch.zeros(input_ids.size(0), pad_size, device=device).long(), input_ids], dim=1)
            inputs['input_ids'] = torch.cat([input_ids, labels], dim=0)
            inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
            del inputs['labels']
        else:
            inputs['input_ids'] = torch.cat([inputs['input_ids'], inputs['input_ids']], dim=0)
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], inputs['attention_mask']], dim=0)
            del inputs['labels']

        pooler_output = model(output_hidden_states=True, return_dict=True, **inputs).hidden_states[-1][:, -1, :]

        if self.use_neg_sentence:
            batch_size = pooler_output.size(0)//3
            pooler_output = torch.stack([pooler_output[:batch_size],
                                         pooler_output[batch_size:2*batch_size],
                                         pooler_output[2*batch_size:]], dim=1)
            z1, z2, z3 = pooler_output[:,0], pooler_output[:,1], pooler_output[:,2]
        else:
            batch_size = pooler_output.size(0)//2
            pooler_output = torch.stack([pooler_output[:batch_size], pooler_output[batch_size:]], dim=1)
            z1, z2 = pooler_output[:,0], pooler_output[:,1]
        
        # Import Similarity from finetuning
        from tasks.finetuning import Similarity
        loss_fct = torch.nn.CrossEntropyLoss()

        if dist.is_initialized():
            if self.use_neg_sentence:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            # Dummy vectors for allgather
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            # Allgather
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())

            # Since allgather results do not have gradients, we replace the
            # current process's corresponding embeddings with original tensors
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            # Get full batch embeddings: (bs x N, hidden)
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        if not hasattr(model, "sim"):
            self.sim = Similarity(temp=0.05)
        cos_sim = self.sim(z1.unsqueeze(1).float(), z2.unsqueeze(0).float())

        if self.use_neg_sentence:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(device)

        if self.use_neg_sentence:
            z3_weight = 0
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(device)
            cos_sim = cos_sim + weights
        loss = loss_fct(cos_sim, labels)
        return (loss, pooler_output) if return_outputs else loss


def init_distributed():
    """Initialize distributed training environment."""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size
        )
    
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    
    return rank, local_rank, world_size, device


def get_fsdp_config(device):
    """Get FSDP configuration."""
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer}
    )
    
    mixed_precision_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )
    
    return auto_wrap_policy, mixed_precision_policy


def load_model_for_fsdp(llm_path, rank, local_rank, world_size, device):
    """Load model for FSDP training. All ranks load to CPU, then FSDP shards to GPUs."""
    if rank == 0:
        print(f"Loading model from {llm_path}...")
        print("All ranks will load to CPU, then FSDP shards to GPUs (~8.5GB each)")
    
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
            print(f"Rank {rank}: Model loaded")
        dist.barrier(device_ids=[local_rank])
    
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    
    return model, tokenizer


def wrap_model_with_fsdp(model, device, auto_wrap_policy, mixed_precision_policy, cpu_offload_optimizer=False):
    """Wrap model with FSDP.
    
    Args:
        cpu_offload_optimizer: If True, offload optimizer states to CPU to save GPU memory
    """
    # CPU offload for optimizer states (saves ~17GB per GPU)
    cpu_offload = CPUOffload(offload_params=False) if cpu_offload_optimizer else None
    
    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        device_id=device,
        sync_module_states=False,
        use_orig_params=True,  # Required for gradient checkpointing compatibility
        cpu_offload=cpu_offload,  # Offload optimizer states to CPU
        limit_all_gathers=True,
    )
    return model


def get_tokenize_fn(tokenizer, cutoff_len):
    """Create tokenization function."""
    def tokenize(prompt, add_eos_token=True, label_prompt=None, neg_prompt=None):
        result = tokenizer(
            prompt,
            padding=False,
            return_tensors=None,
        )
        
        if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
        ):
            result["input_ids"].append(tokenizer.eos_token_id)
            result["attention_mask"].append(1)
        
        if label_prompt:
            label_result = tokenizer(
                label_prompt,
                padding=False,
                return_tensors=None,
            )
            result["labels"] = label_result["input_ids"]
            if neg_prompt:
                neg_result = tokenizer(
                    neg_prompt,
                    padding=False,
                    return_tensors=None,
                )
                result["attention_mask"] = neg_result["input_ids"]
        else:
            result["labels"] = result["input_ids"].copy()
        
        return result
    
    return tokenize


def get_prompt_fn(tokenizer, cutoff_len, template, use_neg_sentence):
    """Create prompt generation function."""
    tokenize = get_tokenize_fn(tokenizer, cutoff_len)
    
    def generate_and_tokenize_prompt(data_point):
        data_point['input'] = data_point['sent0']
        data_point['output'] = data_point['sent1']
        if use_neg_sentence:
            data_point['neg'] = data_point['hard_neg']
        
        full_prompt = generate_sentemb_prompt(
            data_point, tokenizer, cutoff_len, template, prefix='input'
        )
        pos_full_prompt = generate_sentemb_prompt(
            data_point, tokenizer, cutoff_len, template, prefix='output'
        )
        
        neg_full_prompt = None
        if use_neg_sentence:
            neg_full_prompt = generate_sentemb_prompt(
                data_point, tokenizer, cutoff_len, template, prefix="neg"
            )
        
        tokenized_full_prompt = tokenize(
            full_prompt, 
            False,
            label_prompt=pos_full_prompt,
            neg_prompt=neg_full_prompt
        )
        return tokenized_full_prompt
    
    return generate_and_tokenize_prompt


def load_and_prepare_data(data_path, tokenizer, cutoff_len, template, use_neg_sentence, rank):
    """Load and prepare training data."""
    if rank == 0:
        print(f"Loading data from {data_path}...")
    
    if 'csv' in data_path:
        data = load_dataset("csv", data_files=data_path)
    elif os.path.isdir(data_path):
        data = load_from_disk(data_path)
    else:
        data = load_dataset("json", data_files=data_path)
    
    generate_and_tokenize_prompt = get_prompt_fn(
        tokenizer, cutoff_len, template, use_neg_sentence
    )
    
    if rank == 0:
        print("Tokenizing dataset...")
    
    train_data = data["train"].shuffle().map(
        generate_and_tokenize_prompt, 
        num_proc=8
    )

    if rank == 0:
        print(f"Dataset prepared: {len(train_data)} samples")
    
    return train_data


def train(
    # Model params
    model_name_or_path: str = "/work/piyush/pretrained_checkpoints/Tarsier-34b",
    # Data params
    data_path: str = "data/nli_for_simcse.csv",
    output_dir: str = "./fsdp-output",
    # Training hyperparams
    batch_size: int = 256,
    micro_batch_size: int = 1,  # Reduced default for 34B model
    num_epochs: int = 1,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    cutoff_len: int = 32,
    # Options
    use_neg_sentence: bool = False,
    save_steps: int = 100,
    logging_steps: int = 10,
    seed: int = 42,
    grad_checkpoint: bool = True,
    cpu_offload_optimizer: bool = True,  # Offload optimizer states to CPU to save GPU memory
):
    """Fine-tune a large LLM using FSDP."""
    
    # 1. Initialize Distributed Training
    rank, local_rank, world_size, device = init_distributed()
    
    if rank == 0:
        print("=" * 60)
        print("FSDP Fine-tuning")
        print("=" * 60)
        print(f"World size: {world_size} GPUs")
        print(f"Model: {model_name_or_path}")
        print(f"Data: {data_path}")
        print(f"Output: {output_dir}")
        print("=" * 60)
    
    set_seed(seed)
    
    # 2. Check for LLM weights
    llm_path = model_name_or_path + "-llm"
    if not os.path.exists(llm_path):
        raise RuntimeError(
            f"LLM weights not found at {llm_path}. "
            f"Run `python tasks/split_weights.py -m {model_name_or_path}` first."
        )
    
    # 3. Load Model
    model, tokenizer = load_model_for_fsdp(
        llm_path, rank, local_rank, world_size, device
    )
    
    # Get prompt template - construct directly from EOL_PROMPTS (Tarsier format)
    # For Tarsier: 'USER: {EOL_PROMPTS["text"]} ASSISTANT: '
    template = f'USER: {EOL_PROMPTS["text"]} ASSISTANT: '
    
    if rank == 0:
        print(f"Prompt template: {template[:50]}...")
    
    # 4. Wrap with FSDP (gradient checkpointing will be enabled after wrapping)
    if rank == 0:
        print("\nWrapping model with FSDP...")
    
    auto_wrap_policy, mixed_precision_policy = get_fsdp_config(device)
    model = wrap_model_with_fsdp(
        model, device, auto_wrap_policy, mixed_precision_policy, 
        cpu_offload_optimizer=cpu_offload_optimizer
    )
    
    if rank == 0 and cpu_offload_optimizer:
        print("CPU offload enabled for optimizer states (saves ~17GB GPU memory)")
    
    # Note: Gradient checkpointing will be enabled by TrainingArguments
    # Don't enable it manually - let transformers handle it with FSDP
    
    dist.barrier(device_ids=[local_rank])
    
    if rank == 0:
        gpu_mem = torch.cuda.memory_allocated(device) / 1024**3
        print(f"FSDP wrapped. GPU memory: {gpu_mem:.2f}GB per GPU")
    
    # 5. Load Data
    train_data = load_and_prepare_data(
        data_path, tokenizer, cutoff_len, template, use_neg_sentence, rank
    )
    
    DC_FUN = DataCollatorForSeq2SeqForNeg if use_neg_sentence else transformers.DataCollatorForSeq2Seq
    data_collator = DC_FUN(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
    )
    
    # 6. Setup Trainer
    gradient_accumulation_steps = batch_size // (micro_batch_size * world_size)
    
    if rank == 0:
        print("\nTraining config:")
        print(f"  Global batch size: {batch_size}")
        print(f"  Per-device batch size: {micro_batch_size}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
    
    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        bf16=True,
        logging_steps=logging_steps,
        save_strategy="steps",
        save_steps=save_steps,
        output_dir=output_dir,
        save_total_limit=2,
        load_best_model_at_end=False,
        ddp_find_unused_parameters=False,
        group_by_length=False,
        report_to=None,
        gradient_checkpointing=grad_checkpoint,
        remove_unused_columns=False,
        # Tell Trainer we're using FSDP (even though we wrapped manually)
        # This prevents Trainer from trying to wrap with DDP
        fsdp=["full_shard", "auto_wrap"],
        fsdp_config={
            "min_num_params": 0,
            "xla": False,
            "xla_fsdp_v2": False,
            "xla_fsdp_grad_ckpt": False,
        },
        optim="adamw_bnb_8bit",
    )
    
    trainer = FSDPCompatibleSentembTrainer(
        model=model,
        train_dataset=train_data,
        args=training_args,
        data_collator=data_collator,
    )
    trainer.tokenizer = tokenizer
    trainer.is_nli = True
    trainer.use_neg_sentence = use_neg_sentence
    
    model.config.use_cache = False
    
    # 7. Train
    if rank == 0:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
    
    trainer.train()
    
    # 8. Save
    if rank == 0:
        print("\nSaving model...")
    
    trainer.save_model(output_dir)
    if rank == 0:
        tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")

# Cleanup
    dist.barrier(device_ids=[local_rank])
if dist.is_initialized():
    dist.destroy_process_group()
    

if __name__ == "__main__":
    fire.Fire(train)
