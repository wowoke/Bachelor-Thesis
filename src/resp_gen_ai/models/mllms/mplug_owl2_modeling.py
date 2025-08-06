import torch
from PIL import Image
from transformers import TextStreamer

from .base import MLLMBaseModel
from .mplug_owl2.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from .mplug_owl2.conversation import conv_templates
from .mplug_owl2.mm_utils import (
    KeywordsStoppingCriteria,
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from .mplug_owl2.model.builder import load_pretrained_model


class MLLMmPLUGOwl2(MLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        model_name = get_model_name_from_path(model_path)
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
            model_path, None, model_name, load_8bit=False, load_4bit=False, device=device
        )
        self.device = device

    def __str__(self):
        return "mplug_owl2"

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response.
        """
        image_path = images[0]  # instruct_blip2 only supports single image
        conv = conv_templates["mplug_owl2"].copy()

        image = Image.open(image_path).convert("RGB")
        max_edge = max(
            image.size
        )  # We recommand you to resize to squared image for BEST performance.
        image = image.resize((max_edge, max_edge))

        image_tensor = process_images([image], self.image_processor)
        image_tensor = image_tensor.to(self.device, dtype=torch.float16)
        # test inference
        instruction = instruction[0] if type(instruction) == list else instruction

        inp = DEFAULT_IMAGE_TOKEN + instruction
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = (
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
            .unsqueeze(0)
            .to(self.device)
        )
        stop_str = conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        temperature = 0.7
        max_new_tokens = 200

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
            )

        outputs = self.tokenizer.decode(output_ids[0, input_ids.shape[1] :]).strip()
        return outputs
