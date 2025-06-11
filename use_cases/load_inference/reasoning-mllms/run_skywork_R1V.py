import logging
import argparse
import os

from resp_gen_ai.models import load_rmllm


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

image_path = ["/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/VRP_query_specific/image/ours_full/vrp_146.png","/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/VRP_query_specific/image/ours_full/vrp_146.png"]
instructions = ["描述一下图片","描述一下图片"]


MODEL_PATH = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Skywork--Skywork-R1V-38B/snapshots/c00c3791112a62823098f0721f5a1065314f7076"
model = load_rmllm(testing_model="Skywork-R1V", model_path=MODEL_PATH)

pred = model.generate_batch(
    instructions,
    image_path
)
print(f"image path:\t{image_path}")
print(f"Instruction:\t{instruction}")
print(f"Answer:\t{pred}")
print("-" * 20)