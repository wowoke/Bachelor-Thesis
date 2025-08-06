"""
Refer to https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct
Only for Instruct version not for Qwen2-VL-7B
"""

from PIL import Image
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import torch
from .base import MLLMBaseModel
from resp_gen_ai.models.mllms.qwen_vl_utils import process_vision_info


class MLLMQwen2VLInstruct(MLLMBaseModel):
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
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text= [text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens= 1024,
                                            repetition_penalty = 1.2)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return response
    
    def generate_batch(self,instructions, images):
        messages = []
        for instruction,image in zip(instructions,images):
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            messages.append(message)



        # Preparation for batch inference
        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Batch Inference
        generated_ids = self.model.generate(**inputs,
                                             max_new_tokens=1024,
                                             repetition_penalty = 1.2)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts
    