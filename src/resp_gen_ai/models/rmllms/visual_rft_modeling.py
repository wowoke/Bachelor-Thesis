import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
import os
from PIL import Image
import logging
from tqdm import tqdm
import re
import math
from .base import MLLMBaseModel
from accelerate import Accelerator

logging.basicConfig(level=logging.INFO)
torch.manual_seed(1234)
img2description = dict()

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)

class Visual_RFT(MLLMBaseModel):
    def __init__(self, model_path, device="auto"):
        super().__init__(model_path, device)
        self.accelerator = Accelerator()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16).eval()
        self.model = self.accelerator.prepare(self.model)
        self.processor = AutoProcessor.from_pretrained(model_path,use_fast = True)
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)

    def generate(self,instruction,img_path):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": f"Output the bounding box in the image corresponding to the instruction: {instruction}. Output the thinking process in <think> </think> and your grouding box. Following \"<think> thinking process </think>\n<answer></answer>)\" format."}
                ]
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        initial_length = inputs['input_ids'].shape[1]
        inputs["processor"] = self.processor
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        response = self.processor.batch_decode(generated_ids[0][initial_length:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        response = "".join(response)
        return response


    def generate_batch(self,instructions,img_paths):

        assert len(img_paths) == len(instructions)
        batch_messages = []
        for img_path, instruction in zip(img_paths, instructions):
            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT 
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_path},
                        {
                            "type": "text", 
                            "text": instruction
                        }
                    ]
                }
            ]
            batch_messages.append(messages)

        batch_texts = []
        for msgs in batch_messages:
            text_prompt = self.processor.apply_chat_template(
                msgs, 
                tokenize=False, 
                add_generation_prompt=True
            )
            batch_texts.append(text_prompt)
        image_inputs, _ = process_vision_info(batch_messages) 
        # image_tensor = torch.load("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/JailbreakMethod/Jailbreaking-Attack-against-Multimodal-Large-Language-Model/useful_1_99_-19.30.pt")

        inputs = self.processor(
            text=batch_texts,
            images= image_inputs, #[image_tensor]*len(batch_texts),
            padding=True,             # 让批次内对齐
            return_tensors="pt",       # 返回 PyTorch 张量
            do_rescale = True,
        ).to(self.accelerator.device)               

        print("model generating")
        with torch.no_grad():
            generated_ids = self.unwrapped_model.generate(**inputs, max_new_tokens=4096,do_sample=False,use_cache =True,no_repeat_ngram_size=10)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_texts