import os
import torch
import torchvision.transforms as T
from PIL import Image
from abc import ABCMeta, abstractmethod
from typing import Dict, List, Optional, Union
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms.v2 import ToPILImage
from transformers import AutoModel, AutoTokenizer

from utils.model import EOL_PROMPTS, load_architectures_from_config, transform_pixel_values


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

base_registry = {}
encoder_registry = {}


class BaseModel(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "ARCHITECTURE"):
            base_registry[cls.ARCHITECTURE] = cls

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        load_llm: bool = False,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        **kwargs,
    ):
        print(f"Loading {cls.__name__} from {model_name_or_path}")
        return cls(model_name_or_path, load_llm=load_llm, device_map=device_map, **kwargs)


class BaseModelForInternVL3(BaseModel):
    # InternVL2/InternVL3 chat checkpoints use InternVLChatModel architecture key.
    ARCHITECTURE = "InternVLChatModel"
    LLM_CLASS = AutoModel
    MLLM_CLASS = AutoModel

    @property
    def text_eol_prompt(self) -> str:
        return f"<|im_start|>user\n{EOL_PROMPTS['text']}<|im_end|><|im_start|>assistant\n"

    @property
    def image_eol_prompt(self) -> str:
        return f"<|im_start|>user\n{EOL_PROMPTS['image']}<|im_end|><|im_start|>assistant\n"

    @property
    def video_eol_prompt(self) -> str:
        return f"<|im_start|>user\n{EOL_PROMPTS['video']}<|im_end|><|im_start|>assistant\n"

    def __init__(
        self,
        model_name_or_path: str,
        load_llm: bool = False,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        **kwargs,
    ):
        model_class = self.LLM_CLASS if load_llm else self.MLLM_CLASS
        model_dtype = kwargs.get("dtype", kwargs.get("torch_dtype", torch.bfloat16))
        self.model = model_class.from_pretrained(
            model_name_or_path,
            torch_dtype=model_dtype,
            low_cpu_mem_usage=kwargs.get("low_cpu_mem_usage", True),
            trust_remote_code=True,
            device_map=device_map,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=False,
        )
        self.tokenizer.model_max_length = 16384

    def build_transform(self, input_size: int):
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def find_closest_aspect_ratio(
        self,
        aspect_ratio: float,
        target_ratios: List[tuple[int, int]],
        width: int,
        height: int,
        image_size: int,
    ) -> tuple[int, int]:
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self,
        image: Image.Image,
        min_num: int = 1,
        max_num: int = 12,
        image_size: int = 448,
        use_thumbnail: bool = False,
    ) -> List[Image.Image]:
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images


class EncodeMixin(metaclass=ABCMeta):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "ARCHITECTURE"):
            encoder_registry[cls.ARCHITECTURE] = cls

    @abstractmethod
    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode_text(self, text: str | List[str]) -> torch.Tensor:
        raise NotImplementedError


class AutoEncoder:
    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        architecture: Optional[str] = None,
        **kwargs,
    ):
        config_path = os.path.join(model_name_or_path, "config.json")
        if architecture is not None:
            model_arch = architecture
            print(
                f"Argument `architecture` of AutoEncoder is not None. "
                f"Overriding model architecture to {model_arch}."
            )
        else:
            model_arch = load_architectures_from_config(config_path)

        if model_arch not in encoder_registry:
            raise ValueError(
                f"Model architecture {model_arch} is not registered. "
                "You can register it by subclassing EncodeMixin and setting ARCHITECTURE."
            )

        if device_map is None:
            if torch.cuda.is_available():
                device_map = "cuda"
                print(f"Argument `device_map` is None. CUDA is detected. Setting device_map={device_map}.")
            else:
                device_map = "cpu"
                print(
                    f"Argument `device_map` is None. CUDA is not detected. Setting device_map={device_map}."
                )

        model_class = encoder_registry[model_arch]
        return model_class.from_pretrained(
            model_name_or_path,
            load_llm=False,
            device_map=device_map,
            **kwargs,
        )


class EncoderForInternVL3(BaseModelForInternVL3, EncodeMixin):
    def encode_text(self, text: str | List[str]) -> torch.Tensor:
        prompt = self.text_eol_prompt
        if isinstance(text, str):
            text = [text]
        prompts = [prompt.replace("<sent>", t) for t in text]

        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id
        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                pixel_values=None,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1,
                do_sample=True,
                eos_token_id=eos_token_id,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
        return outputs.hidden_states[0][-1][:, -1, :]

    def encode_vision(self, pixel_values: torch.Tensor | List[torch.Tensor]) -> torch.Tensor:
        batched_pixel_values = transform_pixel_values(pixel_values)
        transform = self.build_transform(input_size=448)

        img_context_token_id = self.tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        self.model.img_context_token_id = img_context_token_id
        eos_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        output_embs = []
        for batch in batched_pixel_values:
            # batch: (T, C, H, W)
            num_frames = batch.shape[0]
            base_prompt = self.image_eol_prompt if num_frames == 1 else self.video_eol_prompt
            dynamic_preprocess_max_num = 12 if num_frames == 1 else 1

            pixel_values_list = []
            num_patches_list = []
            for frame in batch:
                img = ToPILImage()(frame).convert("RGB")
                tiles = self.dynamic_preprocess(
                    img,
                    image_size=448,
                    use_thumbnail=True,
                    max_num=dynamic_preprocess_max_num,
                )
                tiles = torch.stack([transform(tile) for tile in tiles])
                num_patches_list.append(tiles.shape[0])
                pixel_values_list.append(tiles)

            flat_pixel_values = torch.cat(pixel_values_list).to(
                device=self.model.device,
                dtype=self.model.dtype,
            )
            assert len(flat_pixel_values) == sum(num_patches_list)

            prompt = base_prompt
            if num_frames > 1:
                video_prefix = "".join([f"Frame{i + 1}: <image>\n" for i in range(len(num_patches_list))])
                prompt = prompt.replace("<video>\n", video_prefix)

            for num_patches in num_patches_list:
                image_tokens = (
                    "<img>"
                    + "<IMG_CONTEXT>" * self.model.num_image_token * num_patches
                    + "</img>"
                )
                prompt = prompt.replace("<image>", image_tokens, 1)

            model_inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = model_inputs["input_ids"].to(self.model.device)
            attention_mask = model_inputs["attention_mask"].to(self.model.device)

            with torch.inference_mode():
                outputs = self.model.generate(
                    pixel_values=flat_pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=1,
                    do_sample=True,
                    eos_token_id=eos_token_id,
                    output_hidden_states=True,
                    return_dict_in_generate=True,
                )
            output_embs.append(outputs.hidden_states[0][-1][:, -1, :])

        return torch.cat(output_embs, dim=0)


if __name__ == "__main__":
    from torch.nn.functional import cosine_similarity
    from utils.video import read_frames_decord

    model_path = "/work/piyush/pretrained_checkpoints/InternVL3-8B"
    encoder = AutoEncoder.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    text = "A person is slicing vegetables in a kitchen."
    text_emb = encoder.encode_text(text)
    print(f"Text embedding shape: {text_emb.shape}")

    frames = read_frames_decord(video_path="assets/demo.mp4", num_frames=16)
    vision_emb = encoder.encode_vision(frames.unsqueeze(0))
    print(f"Vision embedding shape: {vision_emb.shape}")
    print(f"Cosine similarity: {cosine_similarity(vision_emb, text_emb)}")
