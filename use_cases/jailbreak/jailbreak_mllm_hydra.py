# This file is same as jailbreak_mllm.py, only this file supports hydra workflow.

import logging
import torch
import os
from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset,SafeBenchDataset
from resp_gen_ai.metrics.jailbreak.evaluate import calculate_asr_by_checking_refusal_words
from resp_gen_ai.models import load_rmllm, load_lm
import resp_gen_ai
from tqdm import tqdm
import pandas as pd
from PIL import Image
import json
import hydra
from omegaconf import DictConfig
logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

@hydra.main(version_base=None, 
            config_path="/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/src/resp_gen_ai/hydra_files", 
            config_name="config")

def main(cfg: DictConfig):

    testing_model = cfg.model.testing_model
    model_path = cfg.model.model_path
    judge_model_path = cfg.judge_model.judge_model_path
    judge_model_name = cfg.judge_model.judge_testing_model
    csv_path = cfg.data_base.csv_path
    image_dir = cfg.data_base.image_path
    judge_model = load_lm(testing_model=judge_model_name ,model_path=judge_model_path)
    model = load_rmllm(testing_model=testing_model, model_path=model_path)

    eval_dataset = SafeBenchDataset(csv_path=csv_path, image_dir=image_dir)
    batch_size = 8
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn,
    )
    
    result = {}
    unsafe = 0
    index = 1
    for batch in tqdm(eval_dataloader):
        questions = batch["question"]
        instructions = batch["instruction"]
        images = batch["image"]
        task_ids = batch["task_id"]
        
        for i in range(len(questions)):
            response = {}
            question = questions[i]
            instruction = instructions[i]
            image = images[i]
            print(image)
            print(i)
            task_id = task_ids[i]
            

            if image is None:
                logger.warning(f"jump over {task_id}, for image_path not found")
                continue
            

            prompt = f"{instruction}\n{question}"
            logger.info(f"input: {prompt}")
            

            resp = model.generate(
                prompt,
                image, 
            )
            
            logger.info(f"answer: {resp}")
            label = judge_model.generate(question,resp)
            label = label.strip()
            if label.startswith("unsafe"):
                response["label"] = "unsafe"
                unsafe +=1
            else:
                response["label"] = "safe"
            response["question"] = prompt
            response["image"] = image
            response["answer"] = resp
            result[f"{index}"] = response
            index +=1
    result["ASR"] = unsafe/(index-1)
    with open(f"/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/{testing_model}_{judge_model_name}.json", "w",encoding= 'utf-8') as f:
        json.dump(result,f,indent =4)
    print("jsonfile generated")


if __name__ == "__main__":
    main()