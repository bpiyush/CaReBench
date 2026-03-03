import os
import math
import torch
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from utils.model import EOL_PROMPTS, load_architectures_from_config
from typing import Dict, Optional, Union, List
from abc import ABCMeta
import copy
from transformers import AutoProcessor, AutoTokenizer


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


class BaseModelForQwen25VL(BaseModel):
    try:
        from transformers import Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration
        ARCHITECTURE = "Qwen2_5_VLForConditionalGeneration"
        LLM_CLASS = Qwen2_5_VLModel
        MLLM_CLASS = Qwen2_5_VLForConditionalGeneration
    except:
        print("Failed to import Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration.")
        ARCHITECTURE = None
        LLM_CLASS = None
        MLLM_CLASS = None

    @property
    def describe_prompt(self):
        return "Describe the video in detail."

    @property
    def text_eol_prompt(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['text']}],
        }]
        return messages
    
    @property
    def image_eol_prompt(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['image']}],
        }]
        return messages
    
    @property
    def video_eol_prompt(self):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": EOL_PROMPTS['video']}],
        }]
        return messages
    
    def video_edit_eol_prompt(self, sentence):
        messages = [{
            "role": "user",
            "content": [{"type": "text", "text": video_edit_eol_prompt(sentence)}],
        }]
        return messages

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



encoder_registry = {}
from abc import abstractmethod

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



def video_edit_eol_prompt(sentence):
    prompt = "Source video: <video>\nEdit instruction: <sent>\n"\
    "Look at the attached video carefully. The provided text is instruction to edit the video. "\
    "Imagine this edit instruction being applied to the provided video frame.\n"\
    "Summarize the resulting edited video in one word:"
    prompt = prompt.replace("<sent>", sentence)
    return prompt


class EncoderForQwen25VL(BaseModelForQwen25VL, EncodeMixin):
    
    def encode_vision(self, video_paths: List[str]) -> torch.Tensor:
        """
        Encodes a list of video paths into a tensor representation.

        Args:
            video_paths: A list of video paths.

        Returns:
            A tensor representation for each video.
        """
        from qwen_vl_utils import process_vision_info
        
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        
        from copy import deepcopy
        base_conversation = self.video_eol_prompt
        conversations = deepcopy(base_conversation * len(video_paths))
        for j, video_path in enumerate(video_paths):
            # Update j'th conversation content
            conversations[j]['content'] = [
                {"type": "video", "video": video_path},
                base_conversation[0]['content'][0],
            ]
        
        prompts = [
            self.processor.apply_chat_template(
                [conversation], tokenize=False, add_generation_prompt=True
            )
            for conversation in conversations
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversations, return_video_kwargs=True,
        )
        
        inputs = self.processor(
            text=prompts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True,
            )
        return output.hidden_states[0][-1][:, -1, :]
    
    def encode_text(self, text: str | List[str]) -> torch.Tensor:
        from copy import deepcopy

        if isinstance(text, str):
            text = [text]
        base_conversation = self.text_eol_prompt
        conversations = deepcopy(base_conversation * len(text))
        for j, t in enumerate(text):
            # Update j'th conversation content
            conversations[j]['content'] = [
                {"type": "text", "text": t},
                base_conversation[0]['content'][0],
            ]
        prompts = [
            self.processor.apply_chat_template(
                [conversation], tokenize=False, add_generation_prompt=True
            )
            for conversation in conversations
        ]
        inputs = self.processor(
            text=prompts,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
        return output.hidden_states[0][-1][:, -1, :]
    
    def encode_vision_text(self, video_path: str, edit_text: str) -> torch.Tensor:
        from qwen_vl_utils import process_vision_info
        video_path = [video_path]
        
        base_conversation = self.video_edit_eol_prompt(edit_text)
        from copy import deepcopy
        conversations = deepcopy(base_conversation * len(video_path))
        for j, video_path in enumerate(video_path):
            # Update j'th conversation content
            conversations[j]['content'] = [
                {"type": "video", "video": video_path},
                base_conversation[0]['content'][0],
            ]
        
        prompts = [
            self.processor.apply_chat_template(
                [conversation], tokenize=False, add_generation_prompt=True
            )
            for conversation in conversations
        ]
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            conversations, return_video_kwargs=True,
        )
        inputs = self.processor(
            text=prompts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=1, output_hidden_states=True, return_dict_in_generate=True)
        return output.hidden_states[0][-1][:, -1, :]