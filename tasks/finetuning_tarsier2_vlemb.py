import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import fire
import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset, RandomSampler
from transformers import Trainer, TrainerCallback, set_seed
from transformers.file_utils import is_datasets_available
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.trainer_utils import has_length
from transformers.utils import logging

from models.modeling_basemodels import AutoBase
from models.tarsier2.dataset.utils import format_one_sample

warnings.simplefilter(action="ignore", category=FutureWarning)
logger = logging.get_logger(__name__)


class ForceTqdmUpdateCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero and (state.global_step % 5 == 0 or state.global_step < 20):
            logger.warning("")


class Similarity(nn.Module):
    def __init__(self, temp: float = 0.05):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class Tarsier2ChiralPairDataset(Dataset):
    def __init__(
        self,
        rows,
        super_processor,
        text_prompt_template: str,
        video_prompt_template: str,
        max_samples: int = -1,
        overfit_num_rows: int = -1,
        overfit_repeat: int = 1,
    ):
        self.rows = rows
        if max_samples and max_samples > 0:
            self.rows = self.rows.select(range(min(max_samples, len(self.rows))))
        self.super_processor = super_processor
        self.text_prompt_template = text_prompt_template
        self.video_prompt_template = video_prompt_template
        if "<sent>" not in self.text_prompt_template:
            raise ValueError("text_prompt_template must contain '<sent>' for text EOL prompting.")
        if "<video>" not in self.video_prompt_template:
            raise ValueError("video_prompt_template must contain '<video>' for video EOL prompting.")
        self.overfit_num_rows = max(0, overfit_num_rows)
        self.overfit_repeat = max(1, overfit_repeat)

        if self.overfit_num_rows > 0:
            keep = min(self.overfit_num_rows, len(self.rows))
            self.rows = self.rows.select(range(keep))

    def _text_prompt_from_caption(self, caption: str) -> str:
        if "<sent>" in self.text_prompt_template:
            return self.text_prompt_template.replace("<sent>", caption)
        return f"{self.text_prompt_template}\n{caption}"

    @staticmethod
    def _unbatch_tensors(sample_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key, value in sample_dict.items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.dim() > 0 and value.size(0) == 1:
                out[key] = value[0].contiguous()
            else:
                out[key] = value.contiguous()
        return out

    def _encode_video_only(self, video_path: str) -> Dict[str, torch.Tensor]:
        sample = format_one_sample(media_file=str(video_path), prompt=self.video_prompt_template)
        sample = self.super_processor(sample)
        return self._unbatch_tensors(sample)

    def _encode_text_only(self, caption: str) -> Dict[str, torch.Tensor]:
        prompt = self._text_prompt_from_caption(str(caption))
        sample = format_one_sample(media_file=None, prompt=prompt)
        sample = self.super_processor(sample)
        return self._unbatch_tensors(sample)

    def __len__(self):
        if self.overfit_num_rows > 0:
            return len(self.rows) * self.overfit_repeat
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index % len(self.rows)]
        return {
            "video0": self._encode_video_only(row["video0"]),
            "text0": self._encode_text_only(row["sent0"]),
            "video1": self._encode_video_only(row["video_hard_neg"]),
            "text1": self._encode_text_only(row["sent_hard_neg"]),
        }


@dataclass
class DataCollatorForTarsier2Pairs:
    pad_token_id: int
    pad_to_multiple_of: Optional[int] = 8

    def _pad_1d(self, values: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(v.size(0) for v in values)
        if self.pad_to_multiple_of:
            m = self.pad_to_multiple_of
            max_len = ((max_len + m - 1) // m) * m
        padded = []
        for v in values:
            if v.size(0) == max_len:
                padded.append(v)
                continue
            pad = torch.full((max_len - v.size(0),), pad_value, dtype=v.dtype)
            padded.append(torch.cat([pad, v], dim=0))
        return torch.stack(padded, dim=0)

    @staticmethod
    def _merge_tensor_list(values: List[torch.Tensor]) -> torch.Tensor:
        try:
            return torch.stack(values, dim=0)
        except RuntimeError:
            return torch.cat(values, dim=0)

    def _collate_pair(self, pair_samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        batched: Dict[str, torch.Tensor] = {}
        keys = pair_samples[0].keys()
        for key in keys:
            values = [x[key] for x in pair_samples]
            if key == "input_ids":
                batched[key] = self._pad_1d(values, self.pad_token_id)
            elif key == "attention_mask":
                batched[key] = self._pad_1d(values, 0)
            elif key == "labels":
                batched[key] = self._pad_1d(values, -100)
            else:
                batched[key] = self._merge_tensor_list(values)
        return batched

    def __call__(self, features: List[Dict[str, Dict[str, torch.Tensor]]]):
        # Cross-pad positives and hard-negatives together so that _concat_batches
        # (which does torch.cat along dim=0) never sees a seq-len mismatch.
        n = len(features)
        batched = {}
        for a_key, b_key in (("text0", "text1"), ("video0", "video1")):
            combined = self._collate_pair([f[a_key] for f in features] + [f[b_key] for f in features])
            batched[a_key] = {k: v[:n] for k, v in combined.items()}
            batched[b_key] = {k: v[n:] for k, v in combined.items()}
        return batched


class Tarsier2VLContrastiveTrainer(Trainer):
    force_tqdm_update = True

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.force_tqdm_update:
            self.add_callback(ForceTqdmUpdateCallback)

        if self.args.group_by_length:
            if is_datasets_available() and hasattr(self.train_dataset, "column_names"):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name="input_ids",
            )
        return RandomSampler(self.train_dataset)

    @staticmethod
    def _pool_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # The collator left-pads (`cat([pad, v])`) so the EOL anchor is at
        # position S-1. Use the rightmost mask=1 index to be robust to either
        # padding side: argmax on the flipped mask returns the first 1 from the
        # right, which we map back into original index space.
        seq_len = attention_mask.size(1)
        rev_first_one = attention_mask.flip(dims=(-1,)).long().argmax(dim=-1)
        last_idx = (seq_len - 1 - rev_first_one).clamp(min=0)
        row_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[row_idx, last_idx]

    @staticmethod
    def _concat_batches(
        a: Dict[str, torch.Tensor], b: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Concatenate two micro-batches along dim=0 (same keys, e.g. video0+video1)."""
        out: Dict[str, torch.Tensor] = {}
        for key in a:
            if key not in b or not isinstance(a[key], torch.Tensor):
                continue
            out[key] = torch.cat([a[key], b[key]], dim=0)
        return out

    @staticmethod
    def _forward_pair(model, pair_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        model_inputs: Dict[str, torch.Tensor] = {}
        for key, value in pair_inputs.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "labels":
                continue
            model_inputs[key] = value
        model_inputs["output_hidden_states"] = True
        model_inputs["return_dict"] = True
        outputs = model(**model_inputs)
        return {
            "hidden_states": outputs.hidden_states[-1],
            "attention_mask": model_inputs["attention_mask"],
        }

    @staticmethod
    def _gather_with_grad(x: torch.Tensor) -> torch.Tensor:
        if not dist.is_initialized():
            return x
        gathered = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, x.contiguous())
        gathered[dist.get_rank()] = x
        return torch.cat(gathered, dim=0)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        video0 = {k: v.to(model.device) for k, v in inputs["video0"].items()}
        text0 = {k: v.to(model.device) for k, v in inputs["text0"].items()}
        video1 = {k: v.to(model.device) for k, v in inputs["video1"].items()}
        text1 = {k: v.to(model.device) for k, v in inputs["text1"].items()}

        bs = video0["input_ids"].shape[0]
        video_cat = self._concat_batches(video0, video1)
        text_cat = self._concat_batches(text0, text1)

        # Gradient-checkpointed ZeRO-2 fires an independent inner-backward per
        # CheckpointFunction, so two grad-enabled model forwards in one step trigger
        # AccumulateGrad twice per param → "already reduced" assertion.
        # Fix: alternate which modality gets gradients each step; the other runs
        # under no_grad so only ONE grad-enabled forward exists per backward pass.
        if not hasattr(self, "_modal_step"):
            self._modal_step = 0
        video_grad = self._modal_step % 2 == 0
        self._modal_step += 1

        if video_grad:
            out_v = self._forward_pair(model, video_cat)
            with torch.no_grad():
                out_t = self._forward_pair(model, text_cat)
        else:
            with torch.no_grad():
                out_v = self._forward_pair(model, video_cat)
            out_t = self._forward_pair(model, text_cat)

        hv, mv = out_v["hidden_states"], out_v["attention_mask"]
        ht, mt = out_t["hidden_states"], out_t["attention_mask"]
        zv0 = self._pool_last_token(hv[:bs], mv[:bs])
        zv1 = self._pool_last_token(hv[bs:], mv[bs:])
        zt0 = self._pool_last_token(ht[:bs], mt[:bs])
        zt1 = self._pool_last_token(ht[bs:], mt[bs:])

        zv0_g = self._gather_with_grad(zv0)
        zt0_g = self._gather_with_grad(zt0)
        zv1_g = self._gather_with_grad(zv1)
        zt1_g = self._gather_with_grad(zt1)

        if not hasattr(self, "sim"):
            self.sim = Similarity(temp=0.05)

        # Build 2x2 per-sample v->t similarity matrix:
        # [[sim(v0,t0), sim(v0,t1)],
        #  [sim(v1,t0), sim(v1,t1)]]
        s00 = self.sim(zv0_g.float(), zt0_g.float())
        s01 = self.sim(zv0_g.float(), zt1_g.float())
        s10 = self.sim(zv1_g.float(), zt0_g.float())
        s11 = self.sim(zv1_g.float(), zt1_g.float())

        logits_v2t = torch.stack(
            [torch.stack([s00, s01], dim=1), torch.stack([s10, s11], dim=1)],
            dim=1,
        )  # [N, 2, 2]
        labels_2 = torch.tensor([0, 1], device=logits_v2t.device, dtype=torch.long).unsqueeze(0).expand(logits_v2t.size(0), -1)
        loss_v2t = nn.CrossEntropyLoss()(logits_v2t.reshape(-1, 2), labels_2.reshape(-1))

        # Symmetric t->v term for stability.
        logits_t2v = torch.stack(
            [torch.stack([s00, s10], dim=1), torch.stack([s01, s11], dim=1)],
            dim=1,
        )  # [N, 2, 2]
        loss_t2v = nn.CrossEntropyLoss()(logits_t2v.reshape(-1, 2), labels_2.reshape(-1))
        loss = 0.5 * (loss_v2t + loss_t2v)

        with torch.no_grad():
            pos = 0.5 * (s00.mean() + s11.mean())
            neg = 0.5 * (s01.mean() + s10.mean())
            self.log(
                {
                    "train/pos_sim": pos.detach().float(),
                    "train/neg_sim": neg.detach().float(),
                    "train/sim_margin": (pos - neg).detach().float(),
                }
            )

        outputs = {
            "zv0": zv0,
            "zt0": zt0,
            "zv1": zv1,
            "zt1": zt1,
            "logits_v2t_2x2": logits_v2t,
        }
        return (loss, outputs) if return_outputs else loss


def _freeze_non_llm(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False

    llm_module = getattr(model, "language_model", None)
    if llm_module is None:
        raise ValueError("Could not find `language_model` module on Tarsier2 model.")
    for p in llm_module.parameters():
        p.requires_grad = True

    frozen_prefixes = ("vision_tower", "multi_modal_projector")
    for name, p in model.named_parameters():
        if name.startswith(frozen_prefixes):
            p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / max(total, 1):.4f}%)")

    bad = [n for n, p in model.named_parameters() if p.requires_grad and n.startswith(frozen_prefixes)]
    if bad:
        raise RuntimeError(f"Found trainable non-LLM parameters in frozen modules: {bad[:5]}")


def _parse_lora_target_modules(lora_target_modules) -> List[str]:
    if isinstance(lora_target_modules, str):
        return [m.strip() for m in lora_target_modules.split(",") if m.strip()]
    return [str(m).strip() for m in lora_target_modules if str(m).strip()]


def _resolve_lora_target_modules(model: nn.Module, target_suffixes: List[str]) -> List[str]:
    llm_module = getattr(model, "language_model", None)
    if llm_module is None:
        raise ValueError("Could not find `language_model` module on Tarsier2 model.")

    targets: List[str] = []
    for name, _ in model.named_modules():
        if not name.startswith("language_model."):
            continue
        if any(name == f"language_model.{suffix}" or name.endswith(f".{suffix}") for suffix in target_suffixes):
            targets.append(name)

    if not targets:
        raise ValueError(
            "No LoRA target modules found under `language_model` for suffixes: "
            f"{target_suffixes}"
        )
    return targets


def _apply_lora_to_llm(
    model: nn.Module,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules,
) -> nn.Module:
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "LoRA finetuning requires `peft`. Install it with `pip install peft` "
            "or from this repo's requirements.txt."
        ) from exc

    target_suffixes = _parse_lora_target_modules(lora_target_modules)
    target_modules = _resolve_lora_target_modules(model, target_suffixes)
    print(f"Applying LoRA to {len(target_modules)} language_model modules matching: {target_suffixes}")

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    bad = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and ("lora_" not in name or not name.startswith("base_model.model.language_model."))
    ]
    if bad:
        raise RuntimeError(f"Found trainable non-LoRA or non-LLM parameters: {bad[:5]}")
    return model


def _load_rows(data_path: str):
    if "csv" in data_path:
        data = load_dataset("csv", data_files=data_path)
    elif os.path.isdir(data_path):
        data = load_from_disk(data_path)
    else:
        data = load_dataset("json", data_files=data_path)
    return data["train"]


def train(
    model_name_or_path: str = "",
    data_path: str = "data/generated-chiral-pairs-v1.csv",
    output_dir: str = "./finetuned-tarsier2-vlemb",
    batch_size: int = 64,
    micro_batch_size: int = 1,
    num_epochs: int = 1,
    learning_rate: float = 2e-6,
    warmup_ratio: float = 0.0,
    lr_scheduler_type: str = "constant",
    group_by_length: bool = False,
    run_name: Optional[str] = None,
    seed: int = 42,
    deepspeed: Optional[str] = None,
    logging_steps: int = 1,
    grad_checkpoint: bool = True,
    set_pad_to_unk: bool = False,
    bf16: bool = True,
    architecture: Optional[str] = None,
    local_rank: int = 0,
    base_config_name: str = "default_config.yaml",
    max_samples: int = -1,
    overfit_num_rows: int = -1,
    overfit_repeat: int = 1,
    dataloader_num_workers: int = 0,
    report_to_wandb: bool = True,
    lora: bool = False,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj",
):
    del local_rank

    gradient_accumulation_steps = max(1, batch_size // micro_batch_size)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    using_deepspeed = bool(deepspeed) and str(deepspeed).strip() not in ("", "none", "None")

    # Pin GPU before any heavy imports / model load (launcher sets LOCAL_RANK).
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0")))
        torch.cuda.set_device(local_rank)

    if ddp:
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)
        # Do NOT call init_process_group when using the DeepSpeed launcher: DeepSpeed + HF
        # Trainer establish the default process group. A second init corrupts the store and
        # downstream collectives (e.g. all_gather in compute_loss) fail with NCCL recv errors.
        if not using_deepspeed and not dist.is_initialized():
            dist.init_process_group("nccl")

    set_seed(seed)

    base_model = AutoBase.from_pretrained(
        model_name_or_path,
        load_llm=False,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        architecture=architecture,
        base_config_name=base_config_name,
    )
    model = base_model.model
    tokenizer = base_model.tokenizer
    processor = base_model.processor
    super_processor = base_model.super_processor

    if set_pad_to_unk and tokenizer.unk_token_id is not None:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if lora:
        model = _apply_lora_to_llm(
            model=model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
        )
    else:
        _freeze_non_llm(model)

    if grad_checkpoint:
        if hasattr(model, "gradient_checkpointing_enable"):
            # use_reentrant=False avoids double gradient reduction on frozen params under DeepSpeed ZeRO
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    rows = _load_rows(data_path)
    train_dataset = Tarsier2ChiralPairDataset(
        rows=rows.shuffle(seed=seed),
        super_processor=super_processor,
        text_prompt_template=base_model.text_eol_prompt,
        video_prompt_template=base_model.video_eol_prompt,
        max_samples=max_samples,
        overfit_num_rows=overfit_num_rows,
        overfit_repeat=overfit_repeat,
    )
    data_collator = DataCollatorForTarsier2Pairs(
        pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )

    trainer = Tarsier2VLContrastiveTrainer(
        model=model,
        train_dataset=train_dataset,
        data_collator=data_collator,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            lr_scheduler_type=lr_scheduler_type,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            fp16=not bf16,
            bf16=bf16,
            logging_steps=logging_steps,
            save_strategy="no",
            output_dir=output_dir,
            # True is required: frozen vision_tower/projector params are in the forward graph
            # but produce no gradients — ZeRO stage 1/2 will raise "already reduced" otherwise.
            ddp_find_unused_parameters=True if ddp else None,
            group_by_length=group_by_length,
            run_name=run_name,
            report_to="wandb" if report_to_wandb else "none",
            deepspeed=deepspeed,
            gradient_checkpointing=grad_checkpoint,
            remove_unused_columns=False,
            dataloader_num_workers=dataloader_num_workers,
            dataloader_drop_last=True,
            max_grad_norm=1.0,
        ),
        callbacks=[ForceTqdmUpdateCallback],
    )
    trainer.tokenizer = tokenizer
    model.config.use_cache = False

    _log_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if _log_rank == 0:
        print(
            "[finetuning_tarsier2_vlemb] optimizer hyperparams: "
            f"learning_rate={learning_rate}, warmup_ratio={warmup_ratio}, "
            f"lr_scheduler_type={lr_scheduler_type}, "
            f"per_device_train_batch_size={micro_batch_size}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}, "
            f"max_grad_norm=1.0",
            flush=True,
        )

    print("Starting training")
    trainer.train()

    # Final-save only policy: exactly one save at the end.
    if lora:
        model.save_pretrained(output_dir)
    else:
        trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    fire.Fire(train)
