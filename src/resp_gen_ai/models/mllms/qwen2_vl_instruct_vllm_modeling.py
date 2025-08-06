from PIL import Image
import PIL
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
import torch
from .base import MLLMBaseModel
from resp_gen_ai.models.mllms.qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams

import os
import argparse
from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset,SafeBenchDataset
from tqdm import tqdm
import pandas as pd
import json
import resp_gen_ai
import torch

class MLLMQwen2VLInstructVLLM(MLLMBaseModel):
    def __init__(self, model_path):
        self.llm = LLM(
            model=model_path,
            tensor_parallel_size=4,
            trust_remote_code=True,
            gpu_memory_utilization=0.85
        )
        self.sampling_params = SamplingParams(
        temperature=0.01,
        top_p=0.001,
        max_tokens=2048,
        repetition_penalty=1.4
    )
        print("ablskdjflskdjflsdjf")

    def generate(self, instructions, images):
        inputs= []
        for question,image in zip(instructions, images):
            image = [PIL.Image.open(image)]

            placeholder = "<|image_pad|>"


            prompt = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                    )

            buffer={
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": image  
                    },
                    }
            inputs.append(buffer)
        outputs = self.llm.generate(inputs,sampling_params=self.sampling_params)
        results = []
        for o in outputs:
            results.append(o.outputs[0].text)
        return results


    