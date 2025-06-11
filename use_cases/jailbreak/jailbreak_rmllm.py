import logging
import torch
import os
from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset,SafeBenchDataset
from resp_gen_ai.metrics.jailbreak.evaluate import calculate_asr_by_checking_refusal_words,strongreject_finetuned
from resp_gen_ai.models import load_rmllm, load_lm
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
    testing_model = "PKU-DS-V"
    model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--PKU-Alignment--Align-DS-V/snapshots/0a5c659e4c9fa78a6e64d2144d081725534599cb"
    judge_model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--qylu4156--strongreject-15k-v1/snapshots/4bd893d32390d2cace4f067dc2e3ef5294fd78a2"
    judge_model = "qylu4156/strongreject-15k-v1"
    csv_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/VRP_query_specific/prompt/full_ours/vrp.csv"
    image_dir = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/VRP_query_specific/image/ours_full"
    jailbreak_method= "VRP"
    
    if judge_model != "qylu4156/strongreject-15k-v1":
        judge_model = load_lm(testing_model=judge_model ,model_path=judge_model_path)


    model = load_rmllm(testing_model=testing_model, model_path=model_path)
    # load dataset
    eval_dataset = SafeBenchDataset(csv_path=csv_path, image_dir=image_dir,jailbreak_method = jailbreak_method)
    batch_size = 8 # 表示每次放进去8组数据，在推理过程中，这8组数据完全独立，互不影响。在训练过程中，这八组数据共同影响一次梯度的计算，对模型进行更新。
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn,
    ) # 此时的dataset 已经被分成batch_size个batch了
    
    result = {}
    unsafe = 0
    index = 1
    questions_list = [] # only for strong reject evaluator
    answers_list = []
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
            if jailbreak_method == "VRP":
                prompt = question
            else:
                prompt = f"{instruction}\n{question}"
            logger.info(f"input: {prompt}")
            

            resp = model.generate(
                prompt,
                image, 
            )
            # next two lines are only for strong reject 
            questions_list.append(question)
            answers_list.append(resp)
            
            logger.info(f"answer: {resp}")
            if judge_model != "qylu4156/strongreject-15k-v1":
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

    with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/final_first_formal_results_without_evaluation.json", "w",encoding= 'utf-8') as f:
        json.dump(result,f,indent =4)

    buffer = {}
    buffer["forbidden_prompt"] = questions_list
    buffer["response"] = answers_list

    with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/question_and_answer_for_strong_reject.json", "w",encoding= 'utf-8') as f:
        json.dump(buffer,f,indent =4)
    print("without evaluation generated!")
    if judge_model != "qylu4156/strongreject-15k-v1":
        result["ASR"] = unsafe/(index-1)
    elif judge_model == "qylu4156/strongreject-15k-v1":
        """
        {
        "forbidden_prompt": [str, str, str, ...],
        "response": [str, str, str, ...]
        }
        """
        #首先把所有的prompt和answer写成strong reject能接受的形式
        batch = {}
        batch["forbidden_prompt"] = questions_list
        batch["response"] = answers_list
        scores = strongreject_finetuned(batch,512,judge_model_path)
        print(scores)
        result["strong_reject_evaluator"] = scores

    with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/final_first_fomal_result_with_dsv_strong_reject.json", "w",encoding= 'utf-8') as f:
        json.dump(result,f,indent =4)
    print("jsonfile generated!")


if __name__ == "__main__":
    main()