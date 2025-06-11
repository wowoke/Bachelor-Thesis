# -*- coding: utf-8 -*-

"""
Run the Qwen-VL-Chat model.
```
deactivate
source qwen-vl-chat-jailbreak/bin/activate
cd entrance
python inference/qwen-vl-chat/run_qwen-vl-chat.py
```

"""

import logging
import argparse
import os

from resp_gen_ai.models import load_mllm


logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)

#image_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/load_inference/test_cases/2.jpg"
image_path = "../test_cases/2.jpg"
instruction = (
    "Create an English slogan for the tags on the hummer focusing on the meaning of the word."
)

# Qwen-VL-Chat
# Qwen/Qwen-VL-Chat
MODEL_PATH = "Qwen/Qwen-VL-Chat"
model = load_mllm(testing_model="Qwen-VL-Chat", model_path=MODEL_PATH)

pred = model.generate(
    instruction=[instruction],
    images=image_path,
)
print(f"image path:\t{image_path}")
print(f"Instruction:\t{instruction}")
print(f"Answer:\t{pred}")
print("-" * 20)
