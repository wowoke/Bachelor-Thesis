import requests
from PIL import Image
from accelerate import Accelerator
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from .base import MLLMBaseModel

class VLM_R1(MLLMBaseModel):
    def __init__(self, model_path, device="cuda:0"):
        super().__init__(model_path, device)
        # self.accelerator = Accelerator()
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")
        # self.model = self.accelerator.prepare(self.model)
        self.processor = AutoProcessor.from_pretrained(model_path)
        # self.unwrapped_model = self.accelerator.unwrap_model(self.model)

    def generate(self, questions, images):
        