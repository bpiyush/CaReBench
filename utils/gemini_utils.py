import os
import sys
from google import genai
from google.genai import types
import shared.utils as su
import PIL.Image


API_PATH = "/users/piyush/projects/JohanssonBench/secrets/api_key_billing.txt"


class GeminiWrapper:
    """A class to implement zero-shot and few-shot video prompting."""
    def __init__(self, model_key="gemini-2.0-flash", fps=8, secret_key_path=API_PATH, return_response=False):

        self.model_key = model_key
        self.return_response = return_response
        su.log.print_update(f"Loading {model_key} with FPS={fps}", pos="left", fillchar=".")
        self.client = genai.Client(
            api_key=su.io.load_txt(secret_key_path)[0],
        )
        self.fps = fps
    
    def forward_text_only(self, prompt):
        response = self.client.models.generate_content(
            model=f'models/{self.model_key}',
            contents=types.Content(
                parts=[types.Part(text=prompt)]
            )
        )
        if self.return_response:
            return response
        else:
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
        if self.return_response:
            return response
        else:
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
        
        if self.return_response:
            return response
        else:
            return response.text


if __name__ == "__main__":
    # model_key = "gemini-2.0-flash"
    model_key = "gemini-2.5-pro"
    vlm = GeminiWrapper(model_key=model_key, fps=1.)
    video_path = "../JohanssonBench/examples/folding_paper.mp4"
    prompt = "What action is shown in this video?"
    answer = vlm.forward_text_video(prompt, video_path)
    print(answer)
