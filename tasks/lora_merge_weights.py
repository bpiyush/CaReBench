"""Merge weights of a MLLM with a LoRA fine-tuned LLM part of it."""
import os
from models.modeling_basemodels import AutoBase


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', "--base_model", type=str, default="/work/piyush/pretrained_checkpoints/CaRe-7B-Stage-1",
    )
    parser.add_argument(
        '-f', "--fine_tuned_model", type=str, default="/work/piyush/experiments/CaRe/lora/debug_run_nli-27k+ego4d-3k",
    )
    args = parser.parse_args()
    device_map = 'auto'
    
    # Load the MLLM (base model without LLM)
    mllm = AutoBase.from_pretrained(
        args.base_model, load_llm=False, device_map=device_map,
    )

    from peft import PeftModel
    import torch
    from transformers import AutoModel, AutoTokenizer
    
    # Load the base LLM
    llm = AutoBase.from_pretrained(
        args.base_model, load_llm=True, device_map=device_map,
    )
    
    # Load tokenizer from fine-tuned model
    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.fine_tuned_model,
        device_map='auto',
    )
    
    # Load LoRA weights on top of the base LLM
    llm.model = PeftModel.from_pretrained(
        llm.model, args.fine_tuned_model, device_map=device_map,
    )

    # Merge LoRA weights into the base model for inference
    llm.model = llm.model.merge_and_unload()
    
    # Replace the LLM weights in the MLLM with the merged LoRA-finetuned LLM weights
    print("Copying weights of LoRA-finetuned LLM into the LLM part of the MLLM:")
    msg = mllm.model.model.load_state_dict(llm.model.state_dict())
    print(msg)
    
    # Replace the tokenizer in the MLLM with the fine-tuned tokenizer
    print("Replacing the tokenizer in the MLLM with the fine-tuned tokenizer:")
    mllm.tokenizer = llm_tokenizer
    mllm.processor.tokenizer = llm_tokenizer
    
    # Save the merged model
    save_dir = f"{args.fine_tuned_model}/merged_checkpoint"
    os.makedirs(save_dir, exist_ok=True)
    mllm.model.save_pretrained(save_dir)
    mllm.processor.save_pretrained(save_dir)
    mllm.tokenizer.save_pretrained(save_dir)
    print(f"Saved the merged model to {save_dir}")
