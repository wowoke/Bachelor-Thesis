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

image_path = os.path.join(
    os.path.dirname(resp_gen_ai.__file__), "../../use_cases/test_cases/clean.jpeg"
)
instruction = "Describe the input image in detail."
model = load_mllm(testing_model="MiniGPT4-7B", model_path=MODEL_PATH, lm_path=MINIGPT4_7B_VICUNA)
