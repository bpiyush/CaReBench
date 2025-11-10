import os
import sys
from google import genai
from google.genai import types
import shared.utils as su


class GeminiWrapper:
    """A class to implement zero-shot and few-shot video prompting."""
    def __init__(self, model_key="gemini-2.0-flash"):

        self.model_key = model_key
        su.log.print_update(f"Loading {model_key} ", pos="left", fillchar=".")
        REPO_PATH = os.path.expanduser("~/projects/TimeBound.v1/")
        self.client = genai.Client(
            api_key=su.io.load_txt(f"{REPO_PATH}/gemini_api/secrets/api_key_billing.txt")[0],
        )
    
    def generate_answer(self, prompt: str):
        response = self.client.models.generate_content(
            model=f'models/{self.model_key}',
            contents=types.Content(
                parts=[
                    types.Part(text=prompt)
                ]
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
                        inline_data=types.Blob(data=video_bytes, mime_type='video/mp4')
                    ),
                    types.Part(text=prompt)
                ]
            )
        )
        return response.text


if __name__ == "__main__":
    vlm = GeminiWrapper()
    video_path = "sample_data/folding_paper.mp4"
    prompt = "What action is shown in this video?"
    answer = vlm.forward_text_video(prompt, video_path)
    print(answer)
