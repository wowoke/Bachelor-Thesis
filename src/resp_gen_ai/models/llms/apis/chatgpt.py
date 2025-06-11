import base64
import json
import logging
import os
import time
import backoff
import openai

from openai import OpenAI
from openai import OpenAI, OpenAIError
from PIL import Image

import resp_gen_ai

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(resp_gen_ai.__file__), "../../")
KEY_FILE = os.path.join(BASE_DIR, ".key")
KEY = json.load(open(KEY_FILE))["openAI"]
CLIENT = OpenAI(api_key=KEY)
GPT4_LANGUAGE_MODEL = "gpt-4-1106-preview"
GPT4_VISION_MODEL = "gpt-4-vision-preview"


def convert_image_format(input_image):
    TARGET_FORMAT = "png"
    # input_image = "../../datasets/visual_adversarial_example_our/surrogate_instructblip-vicuna-7b/bad_prompt_constrained_16.bmp"
    input_format = input_image.split(".")[-1]
    im = Image.open(input_image)
    rgb_im = im.convert("RGB")
    new_image_path = input_image.replace(input_format, TARGET_FORMAT)
    rgb_im.save(new_image_path)
    logger.info(f"Image saved successfully to {input_image.replace(input_format, TARGET_FORMAT)}")
    return new_image_path


class OpenaiModel:
    def __init__(self, model_name=GPT4_LANGUAGE_MODEL, add_system_prompt=True, sleep=5) -> None:
        self.model_name = model_name
        self.add_system_prompt = add_system_prompt
        self.sleep = sleep

    def fit_message(self, msg):
        if self.add_system_prompt:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": msg},
            ]
        else:
            conversation = [{"role": "user", "content": msg}]
        return conversation

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def inference_language(self, msg, max_tokens=128, **kwargs):
        """
        Perform inference with language model, retrying on exceptions using backoff.
        """
        raw_response = CLIENT.chat.completions.create(
            model=self.model_name,
            messages=self.fit_message(msg),
            max_tokens=max_tokens,
            **kwargs,
        )
        time.sleep(self.sleep)
        return [str(m.message.content) for m in raw_response.choices]


    # Function to encode the image
    @staticmethod
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def fit_message_with_image(self, msg, img_path):
        # Getting the base64 string
        image_format = img_path.split(".")[-1]
        if image_format != "png":
            if os.path.exists(img_path.replace(image_format, "png")):
                img_path = img_path.replace(image_format, "png")
            else:
                logger.info(f"Converting image format from {image_format} to png.")
                img_path = convert_image_format(img_path)
                image_format = img_path.split(".")[-1]
        base64_image = self.encode_image(img_path)
        if self.add_system_prompt:
            conversation = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": msg},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_format};base64,{base64_image}"
                            },
                        },
                    ],
                },
            ]
        else:
            conversation = [{"role": "user", "content": msg}]
        return conversation

    @backoff.on_exception(backoff.expo, OpenAIError, max_time=10, max_tries=6)
    def inference_with_image(self, msg, img_path, max_tokens=128, **kwargs):
        """
        Perform inference with image model, retrying on exceptions using backoff.
        """
        raw_response = CLIENT.chat.completions.create(
            model=self.model_name,
            messages=self.fit_message_with_image(msg, img_path),
            max_tokens=max_tokens,
            **kwargs,
        )
        time.sleep(self.sleep)
        return [str(m.message.content) for m in raw_response.choices]

    class OpenaiModel:
        # ... existing code ...

        def generate(
                self,
                instruction,
                image_path=None,
                max_new_tokens=300,
                top_p=0.9,
                repetition_penalty=1.05,
                length_penalty=1,
                temperature=1.0,
                num_return_sequences=1,
                stop=None,
        ):
            """
            instruction: (str) a string of instruction
            image_path: (str or None) optional path to an image file
            Return: (str) a string of generated response.
            """
            # Prepare additional kwargs for the OpenAI API call
            kwargs = {
                "temperature": temperature,
                "top_p": top_p,
                "n": num_return_sequences,
                "presence_penalty": repetition_penalty,
                "frequency_penalty": length_penalty,
            }

            # Handle stop sequences
            if stop:
                kwargs["stop"] = stop

            # Check if an image is provided
            if image_path:
                # Ensure the image exists
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file '{image_path}' does not exist.")

                # Perform inference with image
                try:
                    response = self.inference_with_image(
                        instruction, img_path=image_path, max_tokens=max_new_tokens, **kwargs
                    )
                except Exception as e:
                    logger.error(f"Error during inference with image: {e}")
                    raise
            else:
                # Perform text-only inference
                try:
                    response = self.inference_language(
                        instruction, max_tokens=max_new_tokens, **kwargs
                    )
                except Exception as e:
                    logger.error(f"Error during text-only inference: {e}")
                    raise

            # Process the response(s)
            if num_return_sequences == 1:
                return response[0].strip()  # Return a single string
            else:
                return [r.strip() for r in response]  # Return a list of responses

def try_one():
    CLIENT.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What’s in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                        },
                    },
                ],
            }
        ],
        max_tokens=300,
    )
