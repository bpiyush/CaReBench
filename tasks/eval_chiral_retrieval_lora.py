"""Evaluate Chiral Retrieval with LoRA fine-tuned models (e.g., Tarsier with LoRA adapters)."""
import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = "False"

import torch
import pandas as pd
import numpy as np
import json
from torch.nn.functional import cosine_similarity
from utils.video import read_frames_decord

import shared.utils as su
from utils.chiral_retrieval_metrics import (
    compute_metrics, print_metrics_as_latex_row,
)


# Constants
DATA_ROOT = "/scratch/shared/beegfs/piyush/datasets"
VIDEO_DIR = {
    "ssv2": f"{DATA_ROOT}/SSv2/20bn-something-something-v2",
    "epic": f"{DATA_ROOT}/EPIC-Kitchens-100/cut_clips",
    "charades": f"{DATA_ROOT}/Charades/Charades_v1_480_cut_clips"
}
EXT = {
    'ssv2': 'webm',
    'epic': 'MP4',
    'charades': 'mp4',
}
REPO_PATH = os.path.expanduser("~/projects/TimeBound.v1/")


def pretty_print_args(args):
    print("========== Arguments ==========")
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("================================\n")


def load_data(dataset='ssv2', split='validation'):
    su.log.print_update(f"Loading data for dataset: {dataset} and split: {split}")

    split_dir = f"{REPO_PATH}/adapt4change/chirality_in_action_splits"
    csv_path = f"{split_dir}/cia-{dataset}-{split}.csv"
    assert os.path.exists(csv_path)
    df = pd.read_csv(csv_path)

    # Add text ID
    df['text_id'] = df[['chiral_triplet_id', 'chiral_label']].apply(
        lambda x: f"{x[0]}_{x[1]}", axis=1,
    )
    video_dir = VIDEO_DIR[dataset]
    ext = EXT[dataset]

    df['video_path'] = df['id'].apply(lambda x: f"{video_dir}/{x}.{ext}")
    df = df[df.video_path.apply(os.path.exists)]
    print("Number of rows: ", len(df))
    print("Sample row: ")
    print(json.dumps(df.iloc[0].to_dict(), indent=4))
    su.log.print_update(f".")
    
    return df


# Define a video processor: video_path -> video_tensor
class VideoProcessor:
    def __init__(self, n_frames=16):
        self.n_frames = n_frames
    
    def __call__(self, video_path):
        video = read_frames_decord(video_path, self.n_frames)
        return video


# Define a feature computer: video_tensor -> video_feature
class VideoFeatureComputer:
    def __init__(self, encoder):
        self.encoder = encoder
    
    def __call__(self, video_tensor):
        with torch.no_grad():
            vision_emb = self.encoder.encode_vision(
                video_tensor.unsqueeze(0),
            ).cpu().squeeze(0).float()
        return vision_emb


# Define a text feature computer: text_str -> text_feature
class TextFeatureComputer:
    def __init__(self, encoder):
        self.encoder = encoder
    
    def __call__(self, text_str):
        with torch.no_grad():
            text_emb = self.encoder.encode_text(text_str).cpu().squeeze(0).float()
        return text_emb


def gather_video_features(df, vfc, vp):
    video_ids = df.id.unique()
    video_feat = {}
    j = 0
    for video_id in su.log.tqdm_iterator(video_ids, desc='Computing video features'):
        video_path = df[df.id == video_id].video_path.unique()[0]
        video_tensor = vp(video_path)
        zv = vfc(video_tensor)
        zv = torch.nn.functional.normalize(zv, dim=-1).float().cpu()
        video_feat[video_id] = zv
        j += 1
    return video_feat


def gather_text_features(df, tfc):
    # Compute text features
    text_ids = df['text_id'].unique()
    texts_feat = {}
    for text_id in su.log.tqdm_iterator(text_ids, desc='Computing text features'):
        text = df[df.text_id == text_id].template.unique()[0]
        zt = tfc(text)
        zt = torch.nn.functional.normalize(zt, dim=-1)
        texts_feat[text_id] = zt.cpu().float()
    return texts_feat


def load_encoder_with_lora(base_model_path, lora_adapter_path, device_map='auto'):
    """
    Load the full MLLM and inject LoRA-finetuned weights into its LLM part.
    
    This function:
    1. Loads the full MLLM (with vision encoder)
    2. Loads the base LLM separately
    3. Applies LoRA adapter to the LLM
    4. Merges LoRA weights
    5. Copies merged LLM weights into the MLLM
    
    Args:
        base_model_path: Path to the base MLLM checkpoint (e.g., Tarsier-7b)
        lora_adapter_path: Path to the LoRA adapter checkpoint
        device_map: Device mapping for model loading
    
    Returns:
        encoder: The encoder with LoRA-finetuned LLM weights
    """
    from models.modeling_basemodels import AutoBase
    from models.modeling_encoders import AutoEncoder
    from peft import PeftModel
    from transformers import AutoTokenizer
    
    print(f"Loading base MLLM from: {base_model_path}")
    print(f"Loading LoRA adapter from: {lora_adapter_path}")
    
    # Load the full MLLM (this will be used for inference)
    mllm = AutoBase.from_pretrained(
        base_model_path, load_llm=False, device_map=device_map,
    )
    
    # Load the base LLM (to apply LoRA on)
    llm = AutoBase.from_pretrained(
        base_model_path, load_llm=True, device_map=device_map,
    )
    
    # Load tokenizer from fine-tuned model (in case it was modified)
    try:
        llm_tokenizer = AutoTokenizer.from_pretrained(
            lora_adapter_path,
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
        llm.model, lora_adapter_path, device_map=device_map,
    )
    
    # Merge LoRA weights into the base model for inference
    print("Merging LoRA weights...")
    llm.model = llm.model.merge_and_unload()
    
    # Copy merged LLM weights into the MLLM's LLM portion
    # The exact attribute depends on the model architecture
    print("Copying LoRA-finetuned LLM weights into MLLM...")
    
    def get_llm_from_mllm(mllm_model):
        """Get the LLM component from an MLLM based on its architecture."""
        if hasattr(mllm_model, 'language_model'):
            return mllm_model.language_model
        elif hasattr(mllm_model, 'model'):
            return mllm_model.model
        elif hasattr(mllm_model, 'llm'):
            return mllm_model.llm
        else:
            raise ValueError("Could not find LLM component in MLLM. Check model architecture.")
    
    mllm_llm = get_llm_from_mllm(mllm.model)
    msg = mllm_llm.load_state_dict(llm.model.state_dict())
    print(f"Loaded state dict: {msg}")
    
    # Update tokenizer
    mllm.tokenizer = llm_tokenizer
    if hasattr(mllm, 'processor') and hasattr(mllm.processor, 'tokenizer'):
        mllm.processor.tokenizer = llm_tokenizer
    
    # Create encoder wrapper
    # We need to wrap the mllm in an encoder interface
    encoder = _create_encoder_from_mllm(mllm, base_model_path)
    
    print("LoRA model loaded successfully!")
    return encoder


def _create_encoder_from_mllm(mllm, base_model_path):
    """
    Create an encoder with encode_vision and encode_text methods from a loaded MLLM.
    
    This wraps the MLLM in an encoder interface compatible with the evaluation code.
    """
    from utils.model import load_architectures_from_config
    from models.modeling_encoders import (
        EncoderForTarsier,
        EncoderForQwen2VL,
        EncoderForCaRe,
    )
    
    # Detect architecture
    config_path = os.path.join(base_model_path, 'config.json')
    arch = load_architectures_from_config(config_path)
    print(f"Detected architecture: {arch}")
    
    # Create an encoder-like wrapper that uses the mllm's model
    class LoRAEncoderWrapper:
        def __init__(self, mllm, arch):
            self.mllm = mllm
            self.model = mllm.model
            self.tokenizer = mllm.tokenizer
            self.processor = mllm.processor
            self.arch = arch
            
            # Import architecture-specific encoding logic
            if 'Tarsier' in arch and 'Tarsier2' not in arch:
                from models.modeling_basemodels import BaseModelForTarsier
                self._base_class = BaseModelForTarsier
            elif 'Tarsier2' in arch:
                from models.modeling_basemodels import BaseModelForTarsier2
                self._base_class = BaseModelForTarsier2
            elif 'Qwen2VL' in arch or 'CaRe' in arch:
                from models.modeling_basemodels import BaseModelForQwen2VL
                self._base_class = BaseModelForQwen2VL
            else:
                self._base_class = None
        
        @property
        def text_eol_prompt(self):
            return self.mllm.text_eol_prompt
        
        @property
        def image_eol_prompt(self):
            return self.mllm.image_eol_prompt
        
        @property
        def video_eol_prompt(self):
            return self.mllm.video_eol_prompt
        
        def encode_vision(self, pixel_values):
            """Encode vision using the appropriate method for this architecture."""
            from utils.model import transform_pixel_values
            from torchvision.transforms.v2 import ToPILImage
            
            pixel_values = transform_pixel_values(pixel_values)
            nframes = pixel_values.shape[1]
            prompt = self.image_eol_prompt if nframes == 1 else self.video_eol_prompt
            
            to_image = ToPILImage()
            batched_frames = []
            for batch in pixel_values:
                frames = [to_image(v) for v in batch]
                batched_frames.append(frames)
            
            generate_kwargs = {
                "max_new_tokens": 1,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
            }
            
            vision_embs = []
            
            for frames in batched_frames:
                input_prompt = prompt.replace("<video>", "<image>" * len(frames))
                input_ids = self.processor.get_text_inputs(input_prompt)
                frames_processed = self.processor.get_pixel_values(frames)
                inputs = {
                    "input_ids": input_ids,
                    "pixel_values": frames_processed
                }
                inputs = {k: v.to(self.model.device) for k, v in inputs.items() if v is not None}
                
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **generate_kwargs)
                vision_embs.append(outputs.hidden_states[0][-1][:, -1, :])
            
            return torch.cat(vision_embs)
        
        def encode_text(self, text):
            """Encode text using the appropriate method for this architecture."""
            prompt = self.text_eol_prompt
            
            if isinstance(text, str):
                text = [text]
            
            prompts = [prompt.replace('<sent>', t) for t in text]
            
            generate_kwargs = {
                "max_new_tokens": 1,
                "output_hidden_states": True,
                "return_dict_in_generate": True,
            }
            
            text_embs = []
            
            for p in prompts:
                text_inputs = self.processor.get_text_inputs(p)
                inputs = {"input_ids": text_inputs}
                inputs = {k: v.to(self.model.device) for k, v in inputs.items() if v is not None}
                
                with torch.inference_mode():
                    outputs = self.model.generate(**inputs, **generate_kwargs)
                text_embs.append(outputs.hidden_states[0][-1][:, -1, :])
            
            return torch.cat(text_embs)
    
    return LoRAEncoderWrapper(mllm, arch)


def load_merged_encoder(merged_model_path, device_map='auto', dtype=torch.bfloat16):
    """
    Load an encoder from a merged checkpoint (created by lora_merge_weights.py).
    
    Args:
        merged_model_path: Path to the merged MLLM checkpoint
        device_map: Device mapping for model loading
        dtype: Data type for model weights
    
    Returns:
        encoder: The encoder ready for inference
    """
    from models.modeling_encoders import AutoEncoder
    
    print(f"Loading merged encoder from: {merged_model_path}")
    encoder = AutoEncoder.from_pretrained(
        merged_model_path,
        device_map=device_map,
        dtype=dtype,
        attn_implementation='flash_attention_2',
    )
    return encoder


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Model loading options (choose one approach)
    parser.add_argument('-m', '--merged_model_path', type=str, default=None,
                        help="Path to merged checkpoint (created by lora_merge_weights.py)")
    parser.add_argument('-b', '--base_model_path', type=str, default=None,
                        help="Path to base MLLM (e.g., Tarsier-7b)")
    parser.add_argument('-l', '--lora_adapter_path', type=str, default=None,
                        help="Path to LoRA adapter checkpoint")
    # Dataset options
    parser.add_argument('-d', '--dataset', type=str, default='ssv2')
    args = parser.parse_args()
    
    pretty_print_args(args)
    
    # Validate arguments
    use_merged = args.merged_model_path is not None
    use_lora = args.base_model_path is not None and args.lora_adapter_path is not None
    
    if not use_merged and not use_lora:
        raise ValueError(
            "Must specify either:\n"
            "  1. --merged_model_path for a merged checkpoint, OR\n"
            "  2. Both --base_model_path and --lora_adapter_path for LoRA loading"
        )
    
    if use_merged and use_lora:
        print("Warning: Both merged and LoRA paths specified. Using merged checkpoint.")
        use_lora = False
    
    # Load data
    df = load_data(dataset=args.dataset, split='validation')
    
    # Load encoder
    if use_merged:
        encoder = load_merged_encoder(args.merged_model_path)
    else:
        encoder = load_encoder_with_lora(args.base_model_path, args.lora_adapter_path)
    
    su.misc.num_params(encoder.model)
    
    vp = VideoProcessor(n_frames=16)
    vfc = VideoFeatureComputer(encoder)
    tfc = TextFeatureComputer(encoder)
    
    video_feat = gather_video_features(df, vfc, vp)
    text_feat = gather_text_features(df, tfc)
    
    metrics = compute_metrics(df, video_feat, text_feat, show_metrics=True)
    print_metrics_as_latex_row(metrics)

