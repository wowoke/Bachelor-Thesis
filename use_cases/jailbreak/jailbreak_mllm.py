import logging
import torch
import os
from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset,SafeBenchDataset
from resp_gen_ai.metrics.jailbreak.evaluate import calculate_asr_by_checking_refusal_words
from resp_gen_ai.models import load_mllm, load_lm
import resp_gen_ai
from tqdm import tqdm
import pandas as pd
from PIL import Image
import json

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch



def main():
    # load models
    testing_model = "Qwen-VL-Chat"
    model_path = "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/.cache/huggingface/hub/models--Qwen--Qwen-VL-Chat/snapshots/f57cfbd358cb56b710d963669ad1bcfb44cdcdd8"
    judge_model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-1B/snapshots/acf7aafa60f0410f8f42b1fa35e077d705892029"
    judge_model = "Llama-Guard-3-1B"
    csv_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/JailbreakMethod/VisualRoleplay/query_specific/prompt/redteam2k/vrp.csv"
    image_dir = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/JailbreakMethod/VisualRoleplay/query_specific/image/redteam2k"
    judge_model = load_lm(testing_model=judge_model ,model_path=judge_model_path)
    model = load_mllm(testing_model=testing_model, model_path=model_path)
    # load dataset
    eval_dataset = SafeBenchDataset(csv_path=csv_path, image_dir=image_dir)
    batch_size = 4 # 表示每次放进去8组数据，在推理过程中，这8组数据完全独立，互不影响。在训练过程中，这八组数据共同影响一次梯度的计算，对模型进行更新。
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate_fn,
    ) # 此时的dataset 已经被分成batch_size个batch了
    
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
            image = images[i]
            print(image)
            print(i)
            task_id = task_ids[i]
            
            if image is None:
                logger.warning(f"jump over {task_id}, for image_path not found.")
                continue
            

            prompt = f"{question}"
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
    with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/VRP_first_try.json", "w",encoding= 'utf-8') as f:
        json.dump(result,f,indent =4)
    print("jsonfile generated!")


if __name__ == "__main__":
    main()