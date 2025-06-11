# -*- coding: utf-8 -*-

"""Calculate ASR on generated responses."""

# 看起来像是给批量实验用的，还不够完善

import logging
import os.path
import time
import json
import pandas as pd

from resp_gen_ai.evaluation.evaluate import (
    calculate_asr_by_checking_refusal_words,
    calculate_asr_by_llama_guard,
)
from resp_gen_ai.models.rmllms.rmllm_inference_zoo import load_rmllm
from resp_gen_ai.models.llms.llm_inference_zoo import load_lm

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

logger = logging.getLogger(__name__)


TASK_TYPES = ["vl", "lm"]

# tmp_dir = "./results/lm"
# ATTACK_METHODS = [
#     dir_name for dir_name in os.listdir(tmp_dir) if os.path.isdir(os.path.join(tmp_dir, dir_name))
# ]
# logger.info(f"attack methods: {ATTACK_METHODS}")

ATTACK_METHODS = [
    "base64",
    "base64_input_only",
    "base64_output_only",
    "base64_raw",
    "combination_1",
    "combination_2",
    "combination_3",
    "disemvowel",
    "leetspeak",
    "rot13"

]

TARGET_MODELS = [
    "Guanaco-7B",
    "Llama2-7B",
    "MPT-7B",
    "Vicuna-7B"
]

SURROGATE_MODELS = [
    "None",
]

BASE_DIR = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/robustness/red-teaming-gpt-4/jailbreak_benchmark/entrance/results/lm"
# MODEL_PATH = "llamas-community/LlamaGuard-7b"
MODEL_PATH = "SebastianSchramm/LlamaGuard-7b-GPTQ-4bit-128g-actorder_True"
model = load_lm(testing_model="Llama-Guard-4bit", model_path=MODEL_PATH)
# model = load_lm(testing_model="Llama-Guard", model_path=MODEL_PATH)

for surrogate_model in SURROGATE_MODELS:
    result_table = pd.DataFrame(
        columns=["Target Model"] + ATTACK_METHODS,
    )
    for target_model in TARGET_MODELS:
        new_row = {"Target Model": target_model}
        for method in ATTACK_METHODS:
            target_model_base_dir = os.path.join(BASE_DIR, method, target_model, surrogate_model)
            if os.path.exists(target_model_base_dir):
                # print(surrogate_base_dir)
                for file in os.listdir(target_model_base_dir):
                    if file.endswith(".json") and file.startswith("results-"):
                        with open(os.path.join(target_model_base_dir, file), "r") as f:
                            data = json.load(f)
                        responses = [(p["instruction_with_behavior"], p["response"]) for p in data["pairs"]]
                        if len(responses) < 1467:
                            continue
                        logger.info(f"computing ASR for target model {target_model} and attack method {method}"
                                    f" with surrogate model {surrogate_model}")
                        time_start = time.time()
                        asr, label_list = calculate_asr_by_llama_guard(responses, model, return_judge=True, saving_tmp=True)
                        for i, d in enumerate(data["pairs"]):
                            d["judge_by_llama_guard"] = label_list[i]
                        with open(os.path.join(target_model_base_dir, "judge-by-llama-guard.json"), "w") as f:
                            json.dump(data, f, indent=4)
                        time_end = time.time()
                        new_row[method] = f"{asr:.4f}"
                        logger.info(f"ASR: {asr:.4f}; time spent: {time_end - time_start}")
        # insert row into the DataFrame, do not use append
        result_table = result_table._append(new_row, ignore_index=True)
    result_table.to_csv(f"llama-guard-4bit-asr-lm-function-{surrogate_model}.csv", index=False)
    logger.info(
        f"ASR table saved llama-guard-asr-lm-function-{surrogate_model}.csv; time spent: {time_end - time_start}"
    )
