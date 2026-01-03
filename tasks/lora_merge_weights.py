"""Merge weights of a MLLM with a LoRA fine-tuned LLM part of it."""
import os
from models.modeling_basemodels import AutoBase
from utils.model import load_architectures_from_config


def get_llm_from_mllm(mllm_model, arch):
    """
    Get the LLM component from an MLLM based on its architecture.
    
    Different VLM architectures store the LLM in different attributes:
    - Tarsier, LlavaNextVideo, InternVL2: model.language_model
    - Qwen2-VL, CaRe: model.model
    - MiniCPM-V: model.llm
    """
    if hasattr(mllm_model, 'language_model'):
        return mllm_model.language_model
    elif hasattr(mllm_model, 'model'):
        return mllm_model.model
    elif hasattr(mllm_model, 'llm'):
        return mllm_model.llm
    else:
        raise ValueError(
            f"Could not find LLM component in MLLM with architecture {arch}. "
            f"Checked attributes: language_model, model, llm"
        )


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
    
    # Detect architecture
    config_path = os.path.join(args.base_model, 'config.json')
    arch = load_architectures_from_config(config_path)
    print(f"Detected architecture: {arch}")
    
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
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(
            args.fine_tuned_model,
            device_map='auto',
        )
        print("Loaded tokenizer from LoRA checkpoint")
    except Exception as e:
        print(f"Could not load tokenizer from LoRA checkpoint: {e}")
        print("Using tokenizer from base model")
        llm_tokenizer = llm.tokenizer
    
    # Load LoRA weights on top of the base LLM
    print("Loading LoRA adapter...")
    llm.model = PeftModel.from_pretrained(
        llm.model, args.fine_tuned_model, device_map=device_map,
    )

    # Merge LoRA weights into the base model for inference
    print("Merging LoRA weights...")
    llm.model = llm.model.merge_and_unload()
    
    # Replace the LLM weights in the MLLM with the merged LoRA-finetuned LLM weights
    print("Copying weights of LoRA-finetuned LLM into the LLM part of the MLLM:")
    mllm_llm = get_llm_from_mllm(mllm.model, arch)
    msg = mllm_llm.load_state_dict(llm.model.state_dict())
    print(msg)
    
    # Replace the tokenizer in the MLLM with the fine-tuned tokenizer
    print("Replacing the tokenizer in the MLLM with the fine-tuned tokenizer:")
    mllm.tokenizer = llm_tokenizer
    if hasattr(mllm, 'processor') and mllm.processor is not None:
        mllm.processor.tokenizer = llm_tokenizer
    
    # Save the merged model
    save_dir = f"{args.fine_tuned_model}/merged_checkpoint"
    os.makedirs(save_dir, exist_ok=True)
    mllm.model.save_pretrained(save_dir)
    
    # Try to save processor (may fail for custom processors like Tarsier)
    if hasattr(mllm, 'processor') and mllm.processor is not None:
        try:
            mllm.processor.save_pretrained(save_dir)
        except (AttributeError, NotImplementedError) as e:
            print(f"Warning: Could not save processor ({e}). Copying from base model instead.")
            # Copy processor files from base model
            import shutil
            for filename in os.listdir(args.base_model):
                if 'processor' in filename.lower() or filename in ['preprocessor_config.json']:
                    src = os.path.join(args.base_model, filename)
                    dst = os.path.join(save_dir, filename)
                    if os.path.isfile(src):
                        shutil.copy2(src, dst)
                        print(f"  Copied {filename}")
    
    mllm.tokenizer.save_pretrained(save_dir)
    print(f"Saved the merged model to {save_dir}")
