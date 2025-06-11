import requests
from PIL import Image

import torch
from .base import MLLMBaseModel

# 下面三个类都来自 Transformers
# AutoModel 适配你的自定义 InternVLChatModel（通过 trust_remote_code）
# AutoTokenizer 负责纯文本分词（对应 tokenizer_config.json）
# AutoImageProcessor 负责图像处理（对应 preprocessor_config.json）
from transformers import AutoModel, AutoTokenizer, AutoImageProcessor

class Eureka_32b(MLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,    
            device_map="auto"              
)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        self.image_processor = AutoImageProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True  
        )


        self.model.eval()


    def generate(self,text,image_path):

        image = Image.open(image_path).convert("RGB")

        prompt_text = text

    # (A) 先处理图像
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(device=self.model.device, dtype=torch.bfloat16)

    # 这一般会得到 { "pixel_values": tensor([...]) }
        generation_config = {
            "max_new_tokens": 500,
            "do_sample": False,
            "top_p": 0,
            "temperature":0
        }

        response = self.model.chat(
            tokenizer=self.tokenizer,
            pixel_values=pixel_values,
            question=prompt_text,
            generation_config=generation_config
        )

        return response


