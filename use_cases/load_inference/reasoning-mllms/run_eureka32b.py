import logging
import argparse
import os

from resp_gen_ai.models import load_rmllm

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

image_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/demo.jpg"
instruction = "Describe this picture."

MODEL_PATH = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--FanqingM--MM-Eureka-Zero-38B/snapshots/99ab88d1ceace45a1d3225cd1df0cc86a1c8ee35"
model = load_rmllm(testing_model="Eureka_32b", model_path=MODEL_PATH)

pred = model.generate(instruction,image_path) # receive a string and a single image_path
print(f"image path:\t{image_path}")
print(f"Instruction:\t{instruction}")
print(f"Answer:\t{pred}")
print("-" * 20)