import os
import sys
import warnings

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import fire
import torch
import torch.distributed as dist
from datasets import load_dataset, load_from_disk
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, set_seed
from tasks.finetuning import DataCollatorForSeq2SeqForNeg, Similarity, generate_sentemb_prompt, SentembTrainer
from utils.model import EOL_PROMPTS

warnings.simplefilter(action='ignore', category=FutureWarning)


def split_weights_internvl3(mllm_path, llm_path):
    """Split InternVL3 MLLM weights to extract only the language model."""
    if os.path.exists(llm_path):
        print(f'{llm_path} already exists. Skip splitting weights.')
        return
    print('Splitting LLM weights from InternVL3 MLLM.')
    model = AutoModel.from_pretrained(
        mllm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    llm = model.language_model
    tokenizer = AutoTokenizer.from_pretrained(mllm_path, trust_remote_code=True, use_fast=False)
    llm.save_pretrained(llm_path)
    tokenizer.save_pretrained(llm_path)
    del model
    torch.cuda.empty_cache()
    print(f'Saved LLM weights to {llm_path}')


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

    llm_path = model_name_or_path + '-llm'
    split_weights_internvl3(model_name_or_path, llm_path)

    model = AutoModelForCausalLM.from_pretrained(
        llm_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
    )
    tokenizer = AutoTokenizer.from_pretrained(
        llm_path,
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.model_max_length = 16384

    if set_pad_to_unk:
        tokenizer.pad_token_id = tokenizer.unk_token_id

    mask_embedding_sentence_template = f"<|im_start|>user\n{EOL_PROMPTS['text']}<|im_end|><|im_start|>assistant\n"
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
        ),
        data_collator=data_collator,
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
