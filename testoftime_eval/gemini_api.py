import os
import sys
from google import genai
from google.genai import types
import shared.utils as su
import PIL.Image


class GeminiWrapper:
    """A class to implement zero-shot and few-shot video prompting."""
    def __init__(self, model_key="gemini-2.0-flash", fps=8, secret_key_path="/users/piyush/projects/JohanssonBench/secrets/api_key_billing.txt"):

        self.model_key = model_key
        su.log.print_update(f"Loading {model_key} ", pos="left", fillchar=".")
        self.client = genai.Client(
            api_key=su.io.load_txt(secret_key_path)[0],
        )
        self.fps = fps

    def forward_text_only(self, prompt: str):
        """Forward pass for text-only input."""
        response = self.client.models.generate_content(
            model=f'models/{self.model_key}',
            contents=types.Content(
                parts=[types.Part(text=prompt)]
            )
        )
        return response.text

    def forward_text_video(self, prompt, video_path: str):
        assert os.path.exists(video_path), \
            f"Video file {video_path} does not exist."
        video_bytes = open(video_path, 'rb').read()
        
        response = self.client.models.generate_content(
            model=f'models/{self.model_key}',
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                        video_metadata=types.VideoMetadata(fps=self.fps),
                    ),
                    types.Part(text=prompt)
                ]
            )
        )
        return response.text
    
    def forward_text_video_image_to_text(self, prompt, video_path: str, image_path: str):
        """
        Given a video as context, ask a question about an image.
        """
        assert os.path.exists(video_path), \
            f"Video file {video_path} does not exist."
        video_bytes = open(video_path, 'rb').read()
        image_bytes = open(image_path, 'rb').read()
        response = self.client.models.generate_content(
            model=f'models/{self.model_key}',
            contents=types.Content(
                parts=[
                    types.Part(
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4'),
                        video_metadata=types.VideoMetadata(fps=self.fps),
                    ),
                    types.Part(
                        inline_data=types.Blob(data=image_bytes, mime_type='image/png'),
                    ),
                    types.Part(text=prompt)
                ]
            )
        )
        return response.text


if __name__ == "__main__":
    # model_key = "gemini-2.0-flash"
    model_key = "gemini-3-pro-preview"
    vlm = GeminiWrapper(model_key=model_key)
    video_path = "examples/folding_paper.mp4"
    prompt = "What action is shown in this video?"
    answer = vlm.forward_text_video(prompt, video_path)
    print(answer)
