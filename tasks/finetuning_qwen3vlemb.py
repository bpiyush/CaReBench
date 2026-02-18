# ruff: noqa: E402
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

import fire
import torch
import torch.nn as nn
import torch.distributed as dist
import transformers
from dataclasses import dataclass
from datasets import load_dataset, load_from_disk
from transformers import Trainer, set_seed
from transformers.utils import logging
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import has_length
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import RandomSampler
from typing import Dict, Optional

from models.qwen3vl_embedding import Qwen3VLForEmbedding
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor


logger = logging.get_logger(__name__)


def apply_accelerate_compat_patch():
    # Compatibility fix for transformers/accelerate version mismatch.
    try:
        from accelerate import Accelerator
        import inspect

        sig = inspect.signature(Accelerator.unwrap_model)
        if "keep_torch_compile" not in sig.parameters:
            original_unwrap_model = Accelerator.unwrap_model

            def patched_unwrap_model(self, model, keep_torch_compile=False):
                return original_unwrap_model(self, model)

            Accelerator.unwrap_model = patched_unwrap_model
    except (ImportError, AttributeError):
        pass


class ForceTqdmUpdateCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            if state.global_step % 5 == 0 or state.global_step < 20:
                logger.warning("")


class Similarity(nn.Module):
    def __init__(self, temp: float):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


@dataclass
class DataCollatorForTriplets:
    tokenizer: transformers.PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = 8
    return_tensors: str = "pt"

    def _pad_batch(self, input_ids_key: str, attention_mask_key: str, features):
        return self.tokenizer.pad(
            {
                "input_ids": [f[input_ids_key] for f in features],
                "attention_mask": [f[attention_mask_key] for f in features],
            },
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

    def __call__(self, features):
        anchor = self._pad_batch("anchor_input_ids", "anchor_attention_mask", features)
        positive = self._pad_batch("positive_input_ids", "positive_attention_mask", features)
        negative = self._pad_batch("negative_input_ids", "negative_attention_mask", features)

        return {
            "anchor_input_ids": anchor["input_ids"],
            "anchor_attention_mask": anchor["attention_mask"],
            "positive_input_ids": positive["input_ids"],
            "positive_attention_mask": positive["attention_mask"],
            "negative_input_ids": negative["input_ids"],
            "negative_attention_mask": negative["attention_mask"],
        }


class Qwen3VLEmbTripletTrainer(Trainer):
    force_tqdm_update = True

    def _get_train_sampler(self, train_dataset) -> Optional[torch.utils.data.Sampler]:
        if train_dataset is None or not has_length(train_dataset):
            return None
        if self.force_tqdm_update:
            self.add_callback(ForceTqdmUpdateCallback)

        if self.args.group_by_length:
            if is_datasets_available() and hasattr(train_dataset, "column_names"):
                lengths = (
                    train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=train_dataset,
                lengths=lengths,
                model_input_name="anchor_input_ids",
            )
        return RandomSampler(train_dataset)

    @staticmethod
    def _pool_last(hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        flipped_mask = attention_mask.flip(dims=[1])
        last_one_positions = flipped_mask.argmax(dim=1)
        col = attention_mask.shape[1] - last_one_positions - 1
        row = torch.arange(hidden_state.shape[0], device=hidden_state.device)
        return hidden_state[row, col]

    @staticmethod
    def _left_pad_to_len(x: torch.Tensor, target_len: int, pad_value: int) -> torch.Tensor:
        if x.size(1) >= target_len:
            return x
        pad = torch.full(
            (x.size(0), target_len - x.size(1)),
            fill_value=pad_value,
            device=x.device,
            dtype=x.dtype,
        )
        return torch.cat([pad, x], dim=1)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        anchor_ids = inputs["anchor_input_ids"]
        positive_ids = inputs["positive_input_ids"]
        negative_ids = inputs["negative_input_ids"]

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        max_len = max(anchor_ids.size(1), positive_ids.size(1), negative_ids.size(1))

        anchor_ids = self._left_pad_to_len(anchor_ids, max_len, pad_token_id)
        positive_ids = self._left_pad_to_len(positive_ids, max_len, pad_token_id)
        negative_ids = self._left_pad_to_len(negative_ids, max_len, pad_token_id)

        all_input_ids = torch.cat([anchor_ids, positive_ids, negative_ids], dim=0)
        all_attention_mask = (all_input_ids != pad_token_id).long()

        outputs = model(
            input_ids=all_input_ids,
            attention_mask=all_attention_mask,
            return_dict=True,
        )
        pooled = self._pool_last(outputs.last_hidden_state, all_attention_mask)
        batch_size = anchor_ids.size(0)
        z1 = pooled[:batch_size]
        z2 = pooled[batch_size:2 * batch_size]
        z3 = pooled[2 * batch_size:]

        if dist.is_initialized():
            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]

            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())

            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z3_list[dist.get_rank()] = z3

            z1 = torch.cat(z1_list, dim=0)
            z2 = torch.cat(z2_list, dim=0)
            z3 = torch.cat(z3_list, dim=0)

        if not hasattr(self, "sim"):
            self.sim = Similarity(temp=0.05)

        pos_logits = self.sim(z1.unsqueeze(1).float(), z2.unsqueeze(0).float())
        neg_logits = self.sim(z1.unsqueeze(1).float(), z3.unsqueeze(0).float())
        logits = torch.cat([pos_logits, neg_logits], dim=1)

        labels = torch.arange(pos_logits.size(0)).long().to(anchor_ids.device)
        loss = nn.CrossEntropyLoss()(logits, labels)

        outputs = {"anchor_emb": z1, "positive_emb": z2, "negative_emb": z3}
        return (loss, outputs) if return_outputs else loss


def _get_string_value(value):
    if isinstance(value, list):
        return value[0] if len(value) > 0 else ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value) if value is not None else ""


def _build_text_prompt(text: str, processor: Qwen3VLProcessor, instruction: str) -> str:
    messages = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user", "content": [{"type": "text", "text": text}]},
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def train(
    model_name_or_path: str = "",
    data_path: str = "data/nli_for_simcse.csv",
    output_dir: str = "./finetuned-qwen3vlemb",
    batch_size: int = 256,
    micro_batch_size: int = 64,
    num_epochs: int = 1,
    learning_rate: float = 5e-5,
    warmup_ratio: float = 0.1,
    cutoff_len: int = 512,
    group_by_length: bool = False,
    run_name: str = None,
    save_steps: int = 1000,
    seed: int = 42,
    deepspeed: str = None,
    logging_steps: int = 10,
    grad_checkpoint: bool = False,
    set_pad_to_unk: bool = False,
    bf16: bool = True,
    text_instruction: str = "Represent the user's input.",
    anchor_column: str = "sent0",
    positive_column: str = "sent1",
    negative_column: str = "hard_neg",
    local_rank: int = 0,
):
    apply_accelerate_compat_patch()

    gradient_accumulation_steps = batch_size // micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        device_id = rank % torch.cuda.device_count()
        torch.cuda.set_device(torch.device(device_id))

    set_seed(seed)

    torch_dtype = torch.bfloat16 if bf16 else torch.float16
    model = Qwen3VLForEmbedding.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    processor = Qwen3VLProcessor.from_pretrained(model_name_or_path, padding_side="right")
    tokenizer = processor.tokenizer

    if set_pad_to_unk and tokenizer.unk_token_id is not None:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    for param in model.parameters():
        param.requires_grad = False
    for param in model.language_model.parameters():
        param.requires_grad = True

    if grad_checkpoint and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if grad_checkpoint and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    def tokenize_text(text: str) -> Dict[str, list]:
        prompt = _build_text_prompt(text=text, processor=processor, instruction=text_instruction)
        tokenized = tokenizer(
            prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
        )
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }

    def generate_and_tokenize_triplet(data_point):
        anchor_text = _get_string_value(data_point.get(anchor_column, ""))
        positive_text = _get_string_value(data_point.get(positive_column, ""))
        negative_text = _get_string_value(data_point.get(negative_column, ""))

        anchor_tokens = tokenize_text(anchor_text)
        positive_tokens = tokenize_text(positive_text)
        negative_tokens = tokenize_text(negative_text)

        return {
            "anchor_input_ids": anchor_tokens["input_ids"],
            "anchor_attention_mask": anchor_tokens["attention_mask"],
            "positive_input_ids": positive_tokens["input_ids"],
            "positive_attention_mask": positive_tokens["attention_mask"],
            "negative_input_ids": negative_tokens["input_ids"],
            "negative_attention_mask": negative_tokens["attention_mask"],
        }

    if "csv" in data_path:
        data = load_dataset("csv", data_files=data_path)
    elif os.path.isdir(data_path):
        data = load_from_disk(data_path)
    else:
        data = load_dataset("json", data_files=data_path)

    original_columns = data["train"].column_names
    train_data = data["train"].shuffle(seed=seed).map(
        generate_and_tokenize_triplet,
        num_proc=1,
        remove_columns=original_columns,
    )

    data_collator = DataCollatorForTriplets(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    trainer = Qwen3VLEmbTripletTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not bf16,
            bf16=bf16,
            logging_steps=logging_steps,
            save_strategy="steps",
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            run_name=run_name,
            report_to=None,
            deepspeed=deepspeed,
            gradient_checkpointing=grad_checkpoint,
            remove_unused_columns=False,
            dataloader_drop_last=True,
        ),
        data_collator=data_collator,
    )
    trainer.tokenizer = tokenizer
    model.config.use_cache = False

    print("Starting training")
    trainer.train()

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
