import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import fire
import torch
import torch.nn as nn
import torch.distributed as dist
import datasets
from datasets import load_dataset, load_from_disk
import transformers
from transformers import Trainer
from transformers import set_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers.utils import logging
from transformers.trainer_callback import TrainerCallback
from typing import Any, Optional, Union
import numpy as np
from dataclasses import dataclass
from models.modeling_basemodels import AutoBase


logger = logging.get_logger(__name__)

class ForceTqdmUpdateCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # pdsh can't update tqdm, except warning
        if state.is_world_process_zero:
            if state.global_step % 5 == 0 or state.global_step < 20:
                logger.warning('')
@dataclass
class DataCollatorForSeq2SeqForNeg:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        _features = self.tokenizer.pad(
            {'input_ids': [feature['input_ids'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )
        _features['attention_mask'] = self.tokenizer.pad(
            {'input_ids': [feature['attention_mask'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )['input_ids']
        _features['labels'] = self.tokenizer.pad(
            {'input_ids': [feature['labels'] for feature in features]},
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )['input_ids']
        features = _features


        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

from transformers.trainer_utils import has_length
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import (
    LengthGroupedSampler,
)
from torch.utils.data import RandomSampler, SequentialSampler

class SentembTrainer(Trainer):
    force_tqdm_update = True
    fix_attention_mask = False

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.force_tqdm_update:
            self.add_callback(ForceTqdmUpdateCallback)

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )

        return RandomSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        # Batch contrastive loss for composed video retrieval
        # z1: embeddings from (source_caption + edit_instruction) using text_edit_eol_prompt
        # z2: embeddings from target_caption using text_eol_prompt
        input_ids, labels = inputs["input_ids"], inputs["labels"]
        labels[labels < 0] = 0
        
        # padding tensor length
        device = input_ids.device
        if input_ids.size(1) > labels.size(1):
            pad_size = input_ids.size(1) - labels.size(1)
            labels = torch.cat([torch.zeros(labels.size(0), pad_size, device=device, dtype=torch.long), labels], dim=1)
        else:
            pad_size = labels.size(1) - input_ids.size(1)
            input_ids = torch.cat([torch.zeros(input_ids.size(0), pad_size, device=device, dtype=torch.long), input_ids], dim=1)
        
        inputs['input_ids'] = torch.cat([input_ids, labels], dim=0)
        inputs['attention_mask'] = (inputs['input_ids'] > 0).long()
        del inputs['labels']

        pooler_output = model(output_hidden_states=True, return_dict=True, **inputs).hidden_states[-1][:, -1, :]

        batch_size = pooler_output.size(0) // 2
        pooler_output = torch.stack([pooler_output[:batch_size], pooler_output[batch_size:]], dim=1)
        z1, z2 = pooler_output[:, 0], pooler_output[:, 1]  # z1: source+edit, z2: target
        
        loss_fct = nn.CrossEntropyLoss()

        if dist.is_initialized():
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

        if not hasattr(self, "sim"):
            self.sim = Similarity(temp=0.05)
        cos_sim = self.sim(z1.unsqueeze(1).float(), z2.unsqueeze(0).float())

        labels = torch.arange(cos_sim.size(0)).long().to(inputs['input_ids'].device)
        loss = loss_fct(cos_sim, labels)
        return (loss, pooler_output) if return_outputs else loss
    
# class ImgembTrainer(Trainer):
    

def get_string_value(value):
    """Extract string value - handle cases where CSV might return lists or other types."""
    if isinstance(value, list):
        return value[0] if len(value) > 0 else ""
    if isinstance(value, (int, float)):
        return str(value)
    return str(value) if value is not None else ""

def generate_sentemb_prompt(text, tokenizer, cutoff_len, template):
    """Generate prompt by truncating text and inserting into template."""
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    truncated_text = tokenizer.decode(tokens['input_ids'])
    return template.replace('<sent>', truncated_text).strip()

def generate_edit_prompt(source_caption, edit_instruction, tokenizer, cutoff_len, template):
    """
    Generate prompt for composed retrieval using text_edit_eol_prompt.
    Replaces <text> with source_caption and <sent> with edit_instruction.
    """
    # Process source_caption
    source_tokens = tokenizer(
        source_caption,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
    source = tokenizer.decode(source_tokens['input_ids'])
    
    # Process edit_instruction
    edit_tokens = tokenizer(
        edit_instruction,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
        add_special_tokens=False,
    )
    edit = tokenizer.decode(edit_tokens['input_ids'])
    
    # Replace both <text> and <sent> in the template
    prompt = template.replace('<text>', source).replace('<sent>', edit).strip()
    return prompt

def train(
    # model/data params
    model_name_or_path: str = "",  # the only required argument
    data_path: str = "data/nli_for_simcse.csv",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 256,
    micro_batch_size: int = 64,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    warmup_ratio: float = 0.2,
    cutoff_len: int = 32,
    # llm hyperparams
    group_by_length: bool = False,
    run_name: str = None,
    use_neg_sentence: bool = False,  # Not used for composed retrieval, kept for compatibility
    save_steps: int = 100,
    seed: int = 42,
    deepspeed: str = None,
    logging_steps: int = 10,
    grad_checkpoint: bool = False,
    fix_attention_mask: bool = False,
    set_pad_to_unk: bool = False,
    bf16: bool = False,
    # make fire happy
    local_rank: int = 0,
):

    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)

    set_seed(seed)

    base_model = AutoBase.from_pretrained(model_name_or_path, load_llm=True, device_map='cuda')
    model = base_model.model
    tokenizer = base_model.tokenizer

    if set_pad_to_unk:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    # Get prompts for composed video retrieval
    text_eol_template = base_model.text_eol_prompt
    # Check if text_edit_eol_prompt exists, otherwise raise error
    if not hasattr(base_model, 'text_edit_eol_prompt'):
        raise ValueError(f"Model {model_name_or_path} does not have text_edit_eol_prompt. "
                        f"Please use a model that supports text editing (e.g., Tarsier).")
    text_edit_eol_template = base_model.text_edit_eol_prompt
    print(f"Text EOL template: {text_eol_template}")
    print(f"Text Edit EOL template: {text_edit_eol_template}")
    
    def tokenize(prompt, add_eos_token=True, label_prompt=None, neg_prompt=None):
        """
        Tokenizes a prompt and returns a result.

        Return:
        {
            "input_ids": List[int],
            "attention_mask": List[int],
            "labels": List[int],
        }
        """
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast

        # tokenizer output format:
        # see https://huggingface.co/docs/datasets/v2.20.0/en/use_dataset#preprocess
        # {'input_ids': [101, 1103, 2067, 1110, 17348, 1106, 1129, 1103, 6880, 1432, 112, 188, 1207, 107, 14255, 1389, 107, 1105, 1115, 1119, 112, 188, 1280, 1106, 1294, 170, 24194, 1256, 3407, 1190, 170, 11791, 5253, 188, 1732, 7200, 10947, 12606, 2895, 117, 179, 7766, 118, 172, 15554, 1181, 3498, 6961, 3263, 1137, 188, 1566, 7912, 14516, 6997, 119, 102], 
        # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
        # 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
        result = tokenizer(
            prompt,
            padding=False,
            return_tensors=None,
        )
        # if token list is not ended with eos token, add eos token
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

    def generate_and_tokenize_prompt(data_point):
        """
        Generates a prompt from a data point and tokenizes it.

        Args:
        data_point: Dict with keys 'source_caption', 'target_caption', 'edit_instruction'
        """
        # Extract string values from data point
        source_caption = get_string_value(data_point.get('source_caption', ''))
        target_caption = get_string_value(data_point.get('target_caption', ''))
        edit_instruction = get_string_value(data_point.get('edit_instruction', ''))
        
        # Generate prompt for source + edit using text_edit_eol_prompt
        edit_prompt = generate_edit_prompt(
            source_caption, edit_instruction, tokenizer, cutoff_len, text_edit_eol_template
        )
        
        # Generate prompt for target using text_eol_prompt
        target_prompt = generate_sentemb_prompt(
            target_caption, tokenizer, cutoff_len, text_eol_template
        )

        tokenized_prompt = tokenize(edit_prompt, False, label_prompt=target_prompt)
        
        # Only return the tokenized fields
        return {
            'input_ids': tokenized_prompt['input_ids'],
            'attention_mask': tokenized_prompt['attention_mask'],
            'labels': tokenized_prompt['labels']
        }
    
    if grad_checkpoint:
        model.enable_input_require_grads()

    if 'csv' in data_path:
        data = load_dataset("csv", data_files=data_path)
    elif os.path.isdir(data_path):
        data = load_from_disk(data_path)
    else:
        data = load_dataset("json", data_files=data_path)

    # model.print_trainable_parameters()  # Be more transparent about the % of trainable params.

    # Get original column names to remove after tokenization
    original_columns = data["train"].column_names

    # train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=25)
    train_data = data["train"].shuffle().map(
        generate_and_tokenize_prompt, 
        num_proc=8,
        remove_columns=original_columns  # Remove original string columns
    )
    # Use standard DataCollatorForSeq2Seq for batch contrastive loss (no hard negatives)
    data_collator = transformers.DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    trainer = SentembTrainer(
        model=model,
        train_dataset=train_data,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=True if not bf16 else False,
            bf16=bf16,
            logging_steps=logging_steps,
            save_strategy="steps",
            eval_steps=None,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=0,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            run_name=run_name,
            report_to=None,
            deepspeed=deepspeed,
            gradient_checkpointing=grad_checkpoint,
            remove_unused_columns=False,
            dataloader_drop_last=True,  # Avoid uneven batches causing NCCL hangs
        ),
        data_collator=data_collator,
    )
    trainer.tokenizer = tokenizer
    trainer.is_nli = True
    trainer.use_neg_sentence = False  # Batch contrastive loss, no hard negatives
    trainer.fix_attention_mask = fix_attention_mask
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    print("Starting training")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(train)
