import logging
import argparse
import os

from resp_gen_ai.models import load_rmllm


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

image_path = ["/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/demo.jpg","/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/image.png"]
instruction = ["1+1等于几?","请做一个自我介绍"]


MODEL_PATH = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--PKU-Alignment--Align-DS-V/snapshots/0a5c659e4c9fa78a6e64d2144d081725534599cb"
model = load_rmllm(testing_model="PKU-DS-V", model_path=MODEL_PATH)

pred = model.generate(
    instructions = instruction,
    images= image_path,
)
print(f"image path:\t{image_path}")
print(f"Instruction:\t{instruction}")
print(f"Answer:\t{pred}")
print("-" * 20)