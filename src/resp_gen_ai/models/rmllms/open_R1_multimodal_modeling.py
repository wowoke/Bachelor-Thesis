from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from .base import MLLMBaseModel
import torch
from PIL import Image
from accelerate import Accelerator

min_pixels = 256*28*28
max_pixels = 1280*28*28

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)
class open_r1_multimodal(MLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        self.accelerator = Accelerator()
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=self.accelerator.device).eval()
        self.model = self.accelerator.prepare(self.model)
        self.processor = AutoProcessor.from_pretrained(model_path,use_fast=True)
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)

    def generate(self, questions, images):
        prompts = []
        for text, image in zip(questions,images):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": f"{image}",
                        },
                        {"type": "text", "text": f"{text}"},
                    ],
                }
            ]
            prompts.append(messages)
        texts = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in prompts]

        image_inputs, video_inputs = process_vision_info(prompts)
        image_tensor = torch.load("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/JailbreakMethod/Jailbreaking-Attack-against-Multimodal-Large-Language-Model/test_jp.pt")
        inputs = self.processor(
            text=texts,
            images= image_inputs,  #[image_tensor] * len(texts),
            padding=True,
            return_tensors="pt",
            do_rescale = True,
        ).to(self.accelerator.device)

        # Inference: Generation of the output
        with torch.no_grad():
            generated_ids = self.unwrapped_model.generate(**inputs, max_new_tokens=4096,max_length=1024, use_cache=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text
