
"""Run the MiniGPT4-13B model."""


import logging
import os

import resp_gen_ai
from resp_gen_ai.models import load_mllm

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

BASE_DIR = os.path.join(os.path.dirname(resp_gen_ai.__file__), "models/mllms/minigpt4_misc/")

image_path = "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/di93zun/zhongyi/RespGenAI-zhongyi/use_cases/load_inference/test_cases/2.jpg"
instruction = (
    "Create an English slogan for the tags on the hummer focusing on the meaning of the word."
)


# 7B to run today
MODEL_PATH = f"{BASE_DIR}/minigpt4_eval_7b.yaml"
MINIGPT4_7B_VICUNA = "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/.cache/huggingface/hub/models--Vision-CAIR--vicuna-7b/snapshots/0abeb8872d75421e58ec146c39b31f5dc78c7613"
model = load_mllm(testing_model="MiniGPT4-7B", model_path=MODEL_PATH, lm_path=MINIGPT4_7B_VICUNA)

pred = model.generate(
    instruction=[instruction],
    images=[image_path],
)

print(f"image path:\t{image_path}")
print(f"Instruction:\t{instruction}")
print(f"Answer:\t{pred}")
print("-" * 20)
