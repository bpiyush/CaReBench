import os
import math
import torch
import torchvision.transforms as T
from PIL import Image
import einops
from torchvision.transforms.v2 import (
    Compose, 
    Resize, 
    CenterCrop, 
    Lambda, 
    ToTensor, 
    Normalize, 
    ToPILImage,
    functional,
)
import math
from torchvision.transforms.functional import InterpolationMode
from transformers import (
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
)
from typing import Dict, Optional, Union, List
from utils.model import EOL_PROMPTS, load_architectures_from_config
from abc import ABCMeta, abstractmethod
import qwen_vl_utils.vision_process as qwen_vl_vision_process

from utils.model import load_architectures_from_config, transform_pixel_values

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
MAX_RATIO = 200

VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = 24576 * 28 * 28
FRAME_FACTOR = 2
FPS = 2.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768


base_registry = {}

class BaseModel(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # register model architecture
        if hasattr(cls, 'ARCHITECTURE'):
            base_registry[cls.ARCHITECTURE] = cls
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        load_llm: bool = False,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        **kwargs):
        print(f'Loading {cls.__name__} from {model_name_or_path}')

        return cls(model_name_or_path, load_llm=load_llm, device_map=device_map, **kwargs)

class AutoBase:
    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        load_llm: bool = False,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        architecture: Optional[str] = None,
        **kwargs):

        config_path = os.path.join(model_name_or_path, 'config.json')
        if architecture is not None:
            model_arch = architecture
            print(f"Argument `architecture` of AutoBase is not None. Overriding model architecture to {model_arch}.")
        else:
            model_arch = load_architectures_from_config(config_path)
        if model_arch not in base_registry:
            raise ValueError(
                f"Model architecture {model_arch} is not registered. "
                "You can register it by subclassing BaseModel and setting ARCHITECTURE attribute."
            )
        if device_map is None:
            if torch.cuda.is_available():
                device_map = 'cuda'
                print(f"Argument `device_map` is None. CUDA is detected. Setting device_map={device_map}.")
            else:
                device_map = 'cpu'
                print(f"Argument `device_map` is None. CUDA is not detected. Setting device_map={device_map}.")
        
        MODEL_CLASS = base_registry[model_arch]

        return MODEL_CLASS.from_pretrained(model_name_or_path, load_llm=load_llm, device_map=device_map, **kwargs)



class BaseModelForQwen3VL(BaseModel):

    ARCHITECTURE = "Qwen3VLForConditionalGeneration"
    LLM_CLASS = Qwen3VLModel
    MLLM_CLASS = Qwen3VLForConditionalGeneration

    @property
    def describe_prompt(self):
        return "Describe the video in detail."

    @property
    def text_eol_prompt(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['text']}],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt
    
    @property
    def image_eol_prompt(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['image']}],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt
    
    @property
    def video_eol_prompt(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['video']}],
        }]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt

    def __init__(
            self, 
            model_name_or_path: str,
            load_llm: Optional[bool] = None,
            device_map: Optional[Union[str, Dict[str, int]]] = None,
            **kwargs,
        ):        
        
        MODEL_CLASS = self.LLM_CLASS if load_llm else self.MLLM_CLASS

        self.load_llm = load_llm

        if load_llm:
            self.split_weights(model_name_or_path, model_name_or_path + '-llm')
            model_name_or_path += '-llm'
        
        self.model = MODEL_CLASS.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
        )
        self.model.eval()
             
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.tokenizer = self.processor.tokenizer

    def split_weights(self, mllm_path, llm_path):
        if os.path.exists(llm_path):
            print(f'{llm_path} already exists. Skip splitting weights.')
            return
        print('Splitting LLM weights from MLLM.')
        model = self.MLLM_CLASS.from_pretrained(mllm_path)
        llm = model.model
        processor = AutoProcessor.from_pretrained(mllm_path)
        tokenizer = AutoTokenizer.from_pretrained(mllm_path)
        llm.save_pretrained(llm_path)
        processor.save_pretrained(llm_path)
        tokenizer.save_pretrained(llm_path)
    
    def round_by_factor(self, number: int, factor: int) -> int:
        """Returns the closest integer to 'number' that is divisible by 'factor'."""
        return round(number / factor) * factor


    def ceil_by_factor(self, number: int, factor: int) -> int:
        """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
        return math.ceil(number / factor) * factor


    def floor_by_factor(self, number: int, factor: int) -> int:
        """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
        return math.floor(number / factor) * factor
    
    def smart_resize(
        self, height: int, width: int, factor: int = None, min_pixels: int = None, max_pixels: int = None
    ) -> tuple[int, int]:
        """
        Rescales the image so that the following conditions are met:

        1. Both dimensions (height and width) are divisible by 'factor'.

        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

        3. The aspect ratio of the image is maintained as closely as possible.
        """
        if max(height, width) / min(height, width) > MAX_RATIO:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
            )
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar


captioner_registry = {}

class CaptionMixin(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # register model architecture
        if hasattr(cls, 'ARCHITECTURE'):
            captioner_registry[cls.ARCHITECTURE] = cls

    def transform_pixel_values(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        # NOTE: this function doesn't accept unbatched inputs
        # pixel_values should be uint8 of (B, T, C, H, W)
        if isinstance(pixel_values, list):
            pixel_values = torch.stack(pixel_values)

        if pixel_values.ndim == 4:
            # pixel_values is (B, C, H, W)
            # (B, C, H, W) -> (B, 1, C, H, W)
            pixel_values = pixel_values.unsqueeze(1)
        elif pixel_values.ndim == 5:
            # pixel_values is (B, T, C, H, W)
            pass
        else:
            raise ValueError(f"pixel_values should be 4D or 5D, got {pixel_values.ndim}D")
        return pixel_values

    @abstractmethod
    def describe(self):
        raise NotImplementedError

class AutoCaptioner:
    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        architecture: Optional[str] = None,
        **kwargs):

        config_path = os.path.join(model_name_or_path, 'config.json')
        if architecture is not None:
            model_arch = architecture
            print(f"Argument `architecture` of AutoEncoder is not None. Overriding model architecture to {model_arch}.")
        else:
            model_arch = load_architectures_from_config(config_path)
        if model_arch not in captioner_registry:
            raise ValueError(
                f"Model architecture {model_arch} is not registered. "
                "You can register it by subclassing EncoderBase and setting ARCHITECTURE attribute."
            )
        if device_map is None:
            if torch.cuda.is_available():
                device_map = 'cuda'
                print(f"Argument `device_map` is None. CUDA is detected. Setting device_map={device_map}.")
            else:
                device_map = 'cpu'
                print(f"Argument `device_map` is None. CUDA is not detected. Setting device_map={device_map}.")
        
        MODEL_CLASS = captioner_registry[model_arch]

        return MODEL_CLASS.from_pretrained(model_name_or_path, load_llm=False, device_map=device_map, **kwargs)


class CaptionerForQwen3VL(BaseModelForQwen3VL, CaptionMixin):
    
    def describe(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> List[str]:
        if self.load_llm:
            raise NotImplementedError("describe method is not implemented for LLM models.")
        
        batched_pixel_values = self.transform_pixel_values(pixel_values)
        descriptions = []
        for pixel_values in batched_pixel_values:
        
            nframes, _, height, width = pixel_values.shape
            min_pixels = qwen_vl_vision_process.VIDEO_MIN_PIXELS
            total_pixels = qwen_vl_vision_process.VIDEO_TOTAL_PIXELS
            max_pixels = max(min(qwen_vl_vision_process.VIDEO_MAX_PIXELS, total_pixels / nframes * qwen_vl_vision_process.FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels = 230400
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=qwen_vl_vision_process.IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            pixel_values = functional.resize(
                pixel_values,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()

            messages = [{
                    "role": "user",
                    "content": [{"type": "text", "text": f"<video>\n{self.describe_prompt}"}],
            }]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ).replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")
            
                    
            inputs = self.processor(
                text=[text],
                images=None,
                videos=[pixel_values],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                generated_ids = self.model.generate(**inputs, max_new_tokens=512, repetition_penalty=1.2)
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            descriptions.append(output_text[0])
        return descriptions


encoder_registry = {}

class EncodeMixin(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # register model architecture
        if hasattr(cls, 'ARCHITECTURE'):
            encoder_registry[cls.ARCHITECTURE] = cls

    @abstractmethod
    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        """
        Encodes vision data (images or videos) into a tensor representation.

        Args:
            pixel_values (torch.Tensor | List[torch.Tensor]): The input pixel values. 
                - If a tensor, it should be of shape (B, C, H, W) for images or (B, T, C, H, W) for videos.
                - If a list, it will be stacked into a tensor.

        Returns:
            torch.Tensor: The encoded tensor representation of the input vision data.

        Raises:
            ValueError: If `pixel_values` is not 4D or 5D.

        ## Notes:
            - This function does not accept unbatched inputs.
            - `pixel_values` should be of type uint8.
        """
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str | List[str]) -> torch.Tensor:
        """
        Encodes the given text(s) into a tensor representation using the model.

        Args:
            text (str | List[str]): A single string or a list of strings to be encoded.

        Returns:
            torch.Tensor: The tensor representation of the encoded text(s).

        ## Notes:
            - The method uses a prompt to encode the text.
            - If a single string is provided, it is converted into a list containing that string.
            - The method processes the prompts and generates the tensor representation using the model.
            - The output tensor contains the hidden states of the last token for each input text.
        """
        raise NotImplementedError

class AutoEncoder:
    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        architecture: Optional[str] = None,
        **kwargs):

        config_path = os.path.join(model_name_or_path, 'config.json')
        if architecture is not None:
            model_arch = architecture
            print(f"Argument `architecture` of AutoEncoder is not None. Overriding model architecture to {model_arch}.")
        else:
            model_arch = load_architectures_from_config(config_path)
        if model_arch not in encoder_registry:
            raise ValueError(
                f"Model architecture {model_arch} is not registered. "
                "You can register it by subclassing EncoderBase and setting ARCHITECTURE attribute."
            )
        if device_map is None:
            if torch.cuda.is_available():
                device_map = 'cuda'
                print(f"Argument `device_map` is None. CUDA is detected. Setting device_map={device_map}.")
            else:
                device_map = 'cpu'
                print(f"Argument `device_map` is None. CUDA is not detected. Setting device_map={device_map}.")
        
        MODEL_CLASS = encoder_registry[model_arch]

        return MODEL_CLASS.from_pretrained(model_name_or_path, load_llm=False, device_map=device_map, **kwargs)
    

class EncoderForQwen3VL(BaseModelForQwen3VL, EncodeMixin):
    
    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        
        batched_pixel_values = transform_pixel_values(pixel_values)
        vision_embs = []
        prompt = self.video_eol_prompt
        prompt = prompt.replace("<video>", "<|vision_start|><|video_pad|><|vision_end|>")

        for pixel_values in batched_pixel_values:
        
            nframes, _, height, width = pixel_values.shape
            min_pixels = VIDEO_MIN_PIXELS
            total_pixels = VIDEO_TOTAL_PIXELS
            max_pixels = max(min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR), int(min_pixels * 1.05))
            max_pixels = 230400
            resized_height, resized_width = self.smart_resize(
                height,
                width,
                factor=IMAGE_FACTOR,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
            pixel_values = functional.resize(
                pixel_values,
                [resized_height, resized_width],
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ).float()

            
            inputs = self.processor(
                text=[prompt],
                images=None,
                videos=[pixel_values],
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.model.device)
            with torch.inference_mode():
                output = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
            vision_embs.append(output.hidden_states[0][-1][:, -1, :])
        vision_embs = torch.cat(vision_embs)
        return vision_embs
    
    def encode_text(self, text: str | List[str]) -> torch.Tensor:

        prompt = self.text_eol_prompt

        if isinstance(text, str):
            text = [text]
        prompts = [prompt.replace('<sent>', t) for t in text]
            
        inputs = self.processor(
            text=prompts,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
        return output.hidden_states[0][-1][:, -1, :]


if __name__ == "__main__":
    model_path = "/work/piyush/pretrained_checkpoints/Qwen3-VL-4B-Instruct"
    captioner = AutoCaptioner.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
    )
    
    from utils.video import read_frames_decord
    frames = read_frames_decord(video_path='assets/demo.mp4', num_frames=32)
    description = captioner.describe(frames.unsqueeze(0))
    print(description[0])
    print('-' * 100)

    from torch.nn.functional import cosine_similarity
    encoder = AutoEncoder.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="auto", attn_implementation="flash_attention_2",
    )
    frames = read_frames_decord(video_path='assets/demo.mp4', num_frames=32)
    text = "This video features a man slicing tomatoes in the kitchen."
    vision_emb = encoder.encode_vision(frames.unsqueeze(0))
    text_emb = encoder.encode_text(text)
    print(f'Vision embedding shape: {vision_emb.shape}')
    print(f'Text embedding shape: {text_emb.shape}')
    print(f'Cosine similarity: {cosine_similarity(vision_emb, text_emb)}')
