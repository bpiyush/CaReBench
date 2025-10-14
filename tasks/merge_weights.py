"""Merge weights of a MLLM with a fine-tuned LLM part of it."""
import os
from models.modeling_basemodels import AutoBase


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-b', "--base_model", type=str, default="/work/piyush/pretrained_checkpoints/CaRe-7B-Stage-1",
    )
    parser.add_argument(
        '-f', "--fine_tuned_model", type=str, default="/work/piyush/experiments/CaRe/debug_run",
    )
    args = parser.parse_args()
    
    mllm = AutoBase.from_pretrained(
        args.base_model, load_llm=False, device_map='auto',
    )
    
    # fine_tuned_model = AutoBase.from_pretrained(
    #     args.base_model, load_llm=True, device_map='auto',
    # )
    
    
    from transformers import AutoModel, AutoTokenizer
    llm = AutoModel.from_pretrained(
        args.fine_tuned_model,
        device_map='auto',
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(
        args.fine_tuned_model,
        device_map='auto',
    )
    
    # Replace the weights of LLM in the MLLM with the fine-tuned LLM weights
    print("Copying weights of finetuned LLM into the LLM part of the MLLM:")
    msg = mllm.model.model.load_state_dict(llm.state_dict())
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

