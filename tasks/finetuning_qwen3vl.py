import os
import sys
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fire
import torch
import torch.nn as nn
import torch.distributed as dist
import datasets
from datasets import load_dataset, load_from_disk
import transformers
from transformers import Trainer, set_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy, logging
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import has_length
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler
from torch.utils.data import RandomSampler
from typing import Any, Optional, Union
import numpy as np
from dataclasses import dataclass

from models.qwen3vl import BaseModelForQwen3VL

warnings.simplefilter(action='ignore', category=FutureWarning)


def apply_accelerate_compat_patch():
    """Compatibility fix for transformers/accelerate version mismatch."""
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


apply_accelerate_compat_patch()

logger = logging.get_logger(__name__)


class ForceTqdmUpdateCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
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

        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
        return features


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


def generate_sentemb_prompt(data_point, tokenizer, cutoff_len, template, prefix='input'):
    sp = f's{prefix}'
    if sp not in data_point:
        input = tokenizer(
            data_point[prefix],
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None,
            add_special_tokens=False,
        )
        input = tokenizer.decode(input['input_ids'])
        data_point[sp] = input
    else:
        input = data_point[sp]
    return template.replace('<sent>', input).strip()


class SentembTrainerForQwen3VL(Trainer):
    force_tqdm_update = True
    fix_attention_mask = False

    @staticmethod
    def _build_text_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
        """Build 3D position IDs for M-ROPE: shape (3, batch_size, seq_len).
        All three components (temporal, height, width) are identical for text."""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids.unsqueeze(0).expand(3, -1, -1)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.is_nli and self.use_neg_sentence:
            input_ids, labels, neg = inputs["input_ids"], inputs["labels"], inputs["attention_mask"]
            pad_token_id = self.tokenizer.pad_token_id
            if self.fix_attention_mask:
                labels[labels < 0] = pad_token_id
                neg[neg < 0] = pad_token_id
            else:
                labels[labels < 0] = 0
                neg[neg < 0] = 0
            mw = max(input_ids.size(1), labels.size(1), neg.size(1))

            pad_size = mw - labels.size(1)
            if pad_size > 0:
                label_pads = torch.zeros(labels.size(0), pad_size, device=input_ids.device).long()
                label_pads.fill_(pad_token_id)
                labels = torch.cat([label_pads, labels], dim=1)
            pad_size = mw - input_ids.size(1)
            if pad_size > 0:
                input_pads = torch.zeros(input_ids.size(0), pad_size, device=input_ids.device).long()
                input_pads.fill_(pad_token_id)
                input_ids = torch.cat([input_pads, input_ids], dim=1)
            pad_size = mw - neg.size(1)
            if pad_size > 0:
                neg_pads = torch.zeros(neg.size(0), pad_size, device=input_ids.device).long()
                neg_pads.fill_(pad_token_id)
                neg = torch.cat([neg_pads, neg], dim=1)

            inputs["input_ids"] = torch.cat([input_ids, labels, neg], dim=0)
            if self.fix_attention_mask:
                inputs["attention_mask"] = (inputs["input_ids"] != pad_token_id).long()
            else:
                inputs["attention_mask"] = (inputs["input_ids"] > 0).long()
            del inputs["labels"]
        elif self.is_nli:
            input_ids, labels = inputs["input_ids"], inputs["labels"]
            labels[labels < 0] = 0
            if input_ids.size(1) > labels.size(1):
                pad_size = input_ids.size(1) - labels.size(1)
                labels = torch.cat([torch.zeros(labels.size(0), pad_size, device=input_ids.device).long(), labels], dim=1)
            else:
                pad_size = labels.size(1) - input_ids.size(1)
                input_ids = torch.cat([torch.zeros(input_ids.size(0), pad_size, device=input_ids.device).long(), input_ids], dim=1)
            inputs["input_ids"] = torch.cat([input_ids, labels], dim=0)
            inputs["attention_mask"] = (inputs["input_ids"] > 0).long()
            del inputs["labels"]
        else:
            inputs["input_ids"] = torch.cat([inputs["input_ids"], inputs["input_ids"]], dim=0)
            inputs["attention_mask"] = torch.cat([inputs["attention_mask"], inputs["attention_mask"]], dim=0)
            del inputs["labels"]

        inputs["position_ids"] = self._build_text_position_ids(inputs["attention_mask"])
        input_ids_for_embeds = inputs["input_ids"]
        model_inputs = {
            "inputs_embeds": model.get_input_embeddings()(input_ids_for_embeds),
            "attention_mask": inputs["attention_mask"],
            "position_ids": inputs["position_ids"],
            "output_hidden_states": True,
            "return_dict": True,
        }
        pooler_output = model(**model_inputs).hidden_states[-1][:, -1, :]

        if self.use_neg_sentence:
            batch_size = pooler_output.size(0) // 3
            pooler_output = torch.stack(
                [pooler_output[:batch_size], pooler_output[batch_size:2 * batch_size], pooler_output[2 * batch_size:]], dim=1
            )
            z1, z2, z3 = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]
        else:
            batch_size = pooler_output.size(0) // 2
            pooler_output = torch.stack([pooler_output[:batch_size], pooler_output[batch_size:]], dim=1)
            z1, z2 = pooler_output[:, 0], pooler_output[:, 1]
        loss_fct = nn.CrossEntropyLoss()

        if dist.is_initialized():
            if self.use_neg_sentence:
                z3_list = [torch.zeros_like(z3) for _ in range(dist.get_world_size())]
                dist.all_gather(tensor_list=z3_list, tensor=z3.contiguous())
                z3_list[dist.get_rank()] = z3
                z3 = torch.cat(z3_list, 0)

            z1_list = [torch.zeros_like(z1) for _ in range(dist.get_world_size())]
            z2_list = [torch.zeros_like(z2) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=z1_list, tensor=z1.contiguous())
            dist.all_gather(tensor_list=z2_list, tensor=z2.contiguous())
            z1_list[dist.get_rank()] = z1
            z2_list[dist.get_rank()] = z2
            z1 = torch.cat(z1_list, 0)
            z2 = torch.cat(z2_list, 0)

        if not hasattr(model, "sim"):
            self.sim = Similarity(temp=0.05)
        cos_sim = self.sim(z1.unsqueeze(1).float(), z2.unsqueeze(0).float())

        if self.use_neg_sentence:
            z1_z3_cos = self.sim(z1.unsqueeze(1), z3.unsqueeze(0))
            cos_sim = torch.cat([cos_sim, z1_z3_cos], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(inputs["input_ids"].device)

        if self.use_neg_sentence:
            z3_weight = 0
            weights = torch.tensor(
                [[0.0] * (cos_sim.size(-1) - z1_z3_cos.size(-1)) + [0.0] * i + [z3_weight] + [0.0] * (z1_z3_cos.size(-1) - i - 1) for i in range(z1_z3_cos.size(-1))]
            ).to(input_ids.device)
            cos_sim = cos_sim + weights
        loss = loss_fct(cos_sim, labels)
        return (loss, pooler_output) if return_outputs else loss


def train(
    model_name_or_path: str = "",
    data_path: str = "data/nli_for_simcse.csv",
    output_dir: str = "./lora-alpaca",
    batch_size: int = 256,
    micro_batch_size: int = 64,
    num_epochs: int = 1,
    learning_rate: float = 5e-4,
    warmup_ratio: float = 0.2,
    cutoff_len: int = 32,
    group_by_length: bool = False,
    run_name: str = None,
    use_neg_sentence: bool = False,
    save_steps: int = 10000,
    seed: int = 42,
    deepspeed: str = None,
    logging_steps: int = 10,
    grad_checkpoint: bool = False,
    fix_attention_mask: bool = False,
    set_pad_to_unk: bool = False,
    bf16: bool = False,
    local_rank: int = 0,
):
    gradient_accumulation_steps = batch_size // micro_batch_size

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size
        torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)

    set_seed(seed)

    base_model = BaseModelForQwen3VL.from_pretrained(
        model_name_or_path, load_llm=False, device_map="cuda"
    )
    model = base_model.model
    tokenizer = base_model.tokenizer

    # Freeze vision encoder — only train the language model
    if hasattr(model, 'visual'):
        for param in model.visual.parameters():
            param.requires_grad = False
        print("Froze vision encoder parameters.")

    if set_pad_to_unk:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    mask_embedding_sentence_template = base_model.text_eol_prompt
    print(mask_embedding_sentence_template)

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

    def generate_and_tokenize_prompt(data_point):
        data_point["input"] = data_point["sent0"]
        data_point["output"] = data_point["sent1"]
        if use_neg_sentence:
            data_point["neg"] = data_point["hard_neg"]

        full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len, mask_embedding_sentence_template, prefix="input")
        pos_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len, mask_embedding_sentence_template, prefix="output")
        if use_neg_sentence:
            neg_full_prompt = generate_sentemb_prompt(data_point, tokenizer, cutoff_len, mask_embedding_sentence_template, prefix="neg")
        else:
            neg_full_prompt = None
        return tokenize(full_prompt, False, label_prompt=pos_full_prompt, neg_prompt=neg_full_prompt)

    if grad_checkpoint:
        model.enable_input_require_grads()

    if "csv" in data_path:
        data = load_dataset("csv", data_files=data_path)
    elif os.path.isdir(data_path):
        data = load_from_disk(data_path)
    else:
        data = load_dataset("json", data_files=data_path)

    train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=8)
    dc_fun = DataCollatorForSeq2SeqForNeg if use_neg_sentence else transformers.DataCollatorForSeq2Seq
    data_collator = dc_fun(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    trainer = SentembTrainerForQwen3VL(
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
        ),
        data_collator=data_collator,
        callbacks=[ForceTqdmUpdateCallback],
    )
    trainer.tokenizer = tokenizer
    trainer.is_nli = True
    trainer.use_neg_sentence = use_neg_sentence
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
