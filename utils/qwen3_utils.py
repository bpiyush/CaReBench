import os
import sys

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.utils.data import Dataset

import shared.utils as su


def load_model(model_name="Qwen/Qwen3-8B", attn_implementation='flash_attention_2'):
    # load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation=attn_implementation,
    )
    su.misc.num_params(model)
    return model, tokenizer


def generate_answer(
    model, tokenizer, prompt, max_new_tokens=2048, enable_thinking=False,
):
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Switches between thinking and non-thinking modes. Default is True.
        enable_thinking=enable_thinking,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 
    
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0
    
    thinking_content = tokenizer.decode(
        output_ids[:index],
        skip_special_tokens=True,
    ).strip("\n")
    content = tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True,
    ).strip("\n")

    return dict(thinking_content=thinking_content, content=content)


class QwenWrapper:
    def __init__(self, model_name="Qwen/Qwen3-8B", enable_thinking=False, attn_implementation='flash_attention_2'):
        su.log.print_update(f"Loading model {model_name}")
        self.model, self.tokenizer = load_model(model_name, attn_implementation)
        self.enable_thinking = enable_thinking
    
    def generate_answer(self, prompt, max_new_tokens=2048):
        """Processes a single prompt at a time."""
        return generate_answer(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens,
            self.enable_thinking,
        )        


class QwenPromptDataset(Dataset):
    """Dataset for batch processing of prompts with Qwen3 model."""
    
    def __init__(self, prompts, tokenizer, enable_thinking=False):
        """
        Initialize dataset with prompts.
        
        Args:
            prompts: List of prompt strings
            tokenizer: Qwen3 tokenizer
            enable_thinking: Whether to enable thinking mode
        """
        self.prompts = prompts
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        """Return tokenized inputs for a single prompt."""
        prompt = self.prompts[idx]
        messages = [{"role": "user", "content": prompt}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        
        # Tokenize and return as dict for collate_fn
        model_inputs = self.tokenizer([text], return_tensors="pt")
        return {
            'input_ids': model_inputs['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': model_inputs['attention_mask'].squeeze(0),
            'prompt_length': len(model_inputs['input_ids'][0])
        }


def collate_qwen_batch(batch):
    """
    Collate function for QwenPromptDataset.
    
    Args:
        batch: List of dicts from __getitem__
    
    Returns:
        Dict with batched inputs ready for model.generate
    """
    # Pad sequences to max length in batch
    max_length = max(item['input_ids'].size(0) for item in batch)
    
    padded_input_ids = []
    padded_attention_mask = []
    prompt_lengths = []
    
    for item in batch:
        current_length = item['input_ids'].size(0)
        padding_length = max_length - current_length
        
        # Pad input_ids with pad_token_id
        padded_input_ids.append(
            torch.cat([
                item['input_ids'],
                torch.full((padding_length,), item['input_ids'][-1], dtype=item['input_ids'].dtype)
            ])
        )
        
        # Pad attention_mask with 0s
        padded_attention_mask.append(
            torch.cat([
                item['attention_mask'],
                torch.zeros(padding_length, dtype=item['attention_mask'].dtype)
            ])
        )
        
        prompt_lengths.append(item['prompt_length'])
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'attention_mask': torch.stack(padded_attention_mask),
        'prompt_lengths': prompt_lengths
    }


def generate_answers_batch(
    model, tokenizer, prompts, max_new_tokens=2048, enable_thinking=False,
    batch_size=4
):
    """
    Generate answers for a batch of prompts efficiently.
    
    Args:
        model: Qwen3 model
        tokenizer: Qwen3 tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum new tokens to generate
        enable_thinking: Whether to enable thinking mode
        batch_size: Batch size for processing
    
    Returns:
        List of dicts with 'thinking_content' and 'content' for each prompt
    """
    from torch.utils.data import DataLoader
    
    # Create dataset and dataloader
    dataset = QwenPromptDataset(prompts, tokenizer, enable_thinking)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        collate_fn=collate_qwen_batch,
        shuffle=False
    )
    
    results = []
    
    for batch in su.log.tqdm_iterator(dataloader, desc="Batch processing"):
        # Move to model device
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Generate
        generated_ids = model.generate(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            max_new_tokens=max_new_tokens,
        )
        
        # Process each sample in the batch
        for i, prompt_length in enumerate(batch['prompt_lengths']):
            # Extract new tokens (after the original prompt)
            output_ids = generated_ids[i][prompt_length:].tolist()
            
            # Parse thinking content
            try:
                # rindex finding 151668 (</think>)
                index = len(output_ids) - output_ids[::-1].index(151668)
            except ValueError:
                index = 0
            
            thinking_content = tokenizer.decode(
                output_ids[:index],
                skip_special_tokens=True,
            ).strip("\n")
            content = tokenizer.decode(
                output_ids[index:],
                skip_special_tokens=True,
            ).strip("\n")
            
            results.append({
                'thinking_content': thinking_content, 
                'content': content
            })
    
    return results


if __name__ == "__main__":
    import shared.utils as su
    import time
    import random
    import string
    
    # Generate 100 random text prompts
    def random_prompt():
        length = random.randint(10, 30)
        return ''.join(random.choices(string.ascii_letters + ' ', k=length))
    
    prompts = [random_prompt() for _ in range(100)]
    
    # Load model and tokenizer
    model, tokenizer = load_model()
    
    # --- Single prompt processing ---
    start = time.time()
    single_results = []
    for prompt in su.log.tqdm_iterator(prompts, desc="Single prompt processing"):
        result = generate_answer(model, tokenizer, prompt)
        single_results.append(result)
    single_time = time.time() - start
    print(f"Single prompt processing: {single_time:.2f} seconds, per sample: {single_time/len(prompts):.4f} s")
    
    # --- Batch processing ---
    start = time.time()
    batch_results = generate_answers_batch(model, tokenizer, prompts, batch_size=8)
    batch_time = time.time() - start
    print(f"Batch processing: {batch_time:.2f} seconds, per sample: {batch_time/len(prompts):.4f} s")
    
    # Optional: check that outputs are similar in structure
    print(f"First single result: {single_results[0]}")
    print(f"First batch result: {batch_results[0]}")
