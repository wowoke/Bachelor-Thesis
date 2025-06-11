
"""
Run the LLaVA  model.
```
deactivate
source llava-jailbreak/bin/activate
cd entrance
python inference/llava/run_llava.py
```.

"""

import logging
import os

import resp_gen_ai
from resp_gen_ai.models.mllms import load_mllm

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(resp_gen_ai.__file__), "models/mllms/minigpt4_misc/")

image_path = os.path.join(
    os.path.dirname(resp_gen_ai.__file__), "../../use_cases/load_inference/test_cases/2.jpg"
)
instruction = (
    "Create an English slogan for the tags on the hummer focusing on the meaning of the word."
)

# LLaVAv1.5-7B
# MODEL_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/red-teaming-gpt-4/jailbreak_benchmark/cache/vlms/llava/llava-v1.5-7b"
# model = load_vlm(testing_model="LLaVAv1.5-7B", model_path=MODEL_PATH, lm_path=None)

# LLaVA-LLaMA2-13B
# MODEL_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/red-teaming-gpt-4/jailbreak_benchmark/cache/vlms/llava/llava-llama-2-13b-chat"
# model = load_vlm(testing_model="LLaVA_llama2-13B", model_path=MODEL_PATH, lm_path=None)

# LLaVA-LLaMA2-7B
# MODEL_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/red-teaming-gpt-4/jailbreak_benchmark/cache/vlms/llava/llava-llama-2-7b-lora"
#MODEL_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--liuhaotian--llava-llama-2-7b-chat-lightning-lora-preview/snapshots/9e8673e2ee6614377ce2d2c71ae15b4258dcf3c8//"
MODEL_PATH = "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/.cache/huggingface/hub/models--llava-hf--llava-v1.6-mistral-7b-hf/snapshots/144bfb964d4eef1502a22af4c5ff20d0d4a94cc1"
#MODEL_PATH = "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-chat-hf/snapshots/f5db02db724555f92da89c216ac04704f23d4590"  
# LoRA 增量权重


model = load_mllm(testing_model="LLaVA_llama2-7B-LoRA", model_path=MODEL_PATH, lm_path=None)


pred = model.generate(
    instruction=[instruction],  
    images=[image_path],
)
