import requests
from PIL import Image
from accelerate import Accelerator
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from .base import MLLMBaseModel

class RMLLMPkuDsV(MLLMBaseModel):
    def __init__(self, model_path, device="auto"):
        super().__init__(model_path, device)
        self.processor = AutoProcessor.from_pretrained(model_path,padding_side = 'left')
        self.accelerator = Accelerator()
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        self.model = self.accelerator.prepare(self.model)
        
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)



    def generate(self, instructions, images):
        """
        批量生成结果的示例。
        :param instructions: List[str]，每条指令文本
        :param images: List[str or PIL.Image.Image]，与 instructions 一一对应的图像
        :return: List[str]，与输入对应的一批结果
        """

        assert len(instructions) == len(images), "instructions 和 images 的长度不匹配"
        all_prompts = []
        all_raw_images = []
        
        for inst, img_path_or_obj in zip(instructions, images):
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{inst}"},
                        {"type": "image"},
                    ],
                }
            ]
            prompt = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True
            )
            all_prompts.append(prompt)

            if isinstance(img_path_or_obj, str):
                raw_image = Image.open(img_path_or_obj)
            else:
                raw_image = img_path_or_obj
            
            all_raw_images.append(raw_image)

        inputs = self.processor(
            images=all_raw_images,
            text=all_prompts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.accelerator.device)

        outputs = self.unwrapped_model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
            use_cache= True
        )
        results = [
            self.processor.decode(output, skip_special_tokens=True)
            for output in outputs
        ]
        return results


