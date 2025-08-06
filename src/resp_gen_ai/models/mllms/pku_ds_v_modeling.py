import requests
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from .base import MLLMBaseModel

class RMLLMPkuDsV(MLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(0)

        self.processor = AutoProcessor.from_pretrained(model_path)

    def generate(self,instruction,images):

        instruction = instruction[0] if type(instruction) == list else instruction
        conversation = [
        {

            "role": "user",
            "content": [
                {"type": "text", "text": f"{instruction}"},
                {"type": "image"},
            ],
        },
        ]
        
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        image_file = images # 这一行是为了适配jailbreak_mllm因为在那里 单张单image已经被提取了
        #PIL对象不能被read 如果是
        if not isinstance(image_file, Image.Image):  
            image_file = images[0]
            raw_image = Image.open(image_file)
        elif isinstance(image_file, Image.Image):  
            raw_image = image_file
        inputs = self.processor(images=raw_image, text=prompt, return_tensors='pt').to(0, torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False)
        output_text = self.processor.decode(output[0], skip_special_tokens=True)
        
        return output_text
