"""
Refer to https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
Only for Instruct version not for Qwen2-VL-7B
"""

from PIL import Image
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import torch
import requests
from .base import MLLMBaseModel
from resp_gen_ai.models.mllms.qwen_vl_utils import process_vision_info


class MLLMQwen2VL(MLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map="auto"
        )
        self.model.eval()

    def __str__(self):
        return "Qwen2VL"

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        
        instruction = instruction[0] if type(instruction) == list else instruction
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": images,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        img_input = messages[0]["content"][0]["image"]
        # 支持本地文件路径或网络 URL
        if isinstance(img_input, str) and (img_input.startswith("http://") or img_input.startswith("https://")):
            img = Image.open(requests.get(img_input, stream=True).raw).convert("RGB")
        elif isinstance(img_input, str):
            img = Image.open(img_input).convert("RGB")
            print("img_type: ",type(img))
        else:
            img = img_input

        # 把对话内容扁平化：直接传图片和文本字典列表给模板
        messages = messages[0]["content"]
        
        # 使用 apply_chat_template 只渲染文本 prompt（现在 messages 是 List[Dict]）
        prompt_str = self.processor.apply_chat_template(
            conversation=messages,
            add_generation_prompt=True,
            tokenize=False        # 只生成带 vision 标记的字符串
        )

        # 不同于Instruct 的是，这里的图片和文本是分开处理的。

        # 将图像和文本一起送入 processor，生成完整 multimodal 输入
        inputs = self.processor(
            images=img,
            text=prompt_str,
            return_tensors="pt"
        ).to(self.model.device)

        # Inference: Generation of the output
        output_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return output_text