"""Merge finetuned LLM weights back into the full InternVL3 MLLM."""
import os
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", "--base_model", type=str, required=True,
        help="Path to the original InternVL3 MLLM checkpoint.",
    )
    parser.add_argument(
        "-f", "--fine_tuned_model", type=str, required=True,
        help="Path to the finetuned LLM checkpoint (output_dir from finetuning).",
    )
    args = parser.parse_args()

    print(f"Loading base InternVL3 MLLM from {args.base_model}")
    mllm = AutoModel.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    print(f"Loading finetuned LLM from {args.fine_tuned_model}")
    llm = AutoModelForCausalLM.from_pretrained(
        args.fine_tuned_model,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.fine_tuned_model,
        trust_remote_code=True,
        use_fast=False,
    )

    print("Copying finetuned LLM weights into the language_model of the MLLM:")
    msg = mllm.language_model.load_state_dict(llm.state_dict())
    print(msg)

    save_dir = os.path.join(args.fine_tuned_model, "merged_checkpoint")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving merged model to {save_dir}")
    mllm.save_pretrained(save_dir)
    llm_tokenizer.save_pretrained(save_dir)
    print(f"Saved merged model to {save_dir}")
