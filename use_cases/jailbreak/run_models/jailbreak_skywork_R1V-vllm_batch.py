from vllm import LLM, SamplingParams
import PIL
import os
import argparse
from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset,REBenchDataset
from tqdm import tqdm
import pandas as pd
import json
import resp_gen_ai
import torch


Jailbreak_Method_and_CSV_PATH= {
    'baseline':"REBench_Dataset/Baseline/REBench_Baseline.csv",
    'base_image':"REBench_Dataset/image.png",

    'cognitive_overload_effect_to_cause': "REBench_Dataset/cognitive_overload/effect_to_cause/REBench_effect_to_cause.csv",
    'effect_to_cause_image': "REBench_Dataset/image.png",

    'cognitive_overload_veiled_attack':"REBench_Dataset/cognitive_overload/veiled_attack/REBench_veiled_attack.csv",
    'veiled_attack_image': "REBench_Dataset/image.png",
    
    'FigStep': "REBench_Dataset/FigStep/REBench_FigStep.csv",
    'FigStep_image_dir': "REBench_Dataset/FigStep/FigStep_Images",

    'VRP': "REBench_Dataset/VRP/REBench_VRP.csv",
    'VRP_image_dir': "REBench_Dataset/VRP/VRP_Images"
}






os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
def save_my_file(result):
    with open("exception.json", "w",encoding= 'utf-8') as f:
        json.dump(result,f,indent =4)
def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

def main():
    batch_size = 16
    device_num = 1
    temperature = 0
    top_p = 0.95
    max_tokens = 4096
    repetition_penalty = 1.4
    MODEL_PATH = "Skywork/Skywork-R1V-38B"


    testing_model = "Skywork_R1V"
    csv_path = "REBench/use_cases/jailbreak/FigStep/REBench_FigStep.csv" # Dataset Path
    images_path = "REBench_Dataset/FigStep/FigStep_Images"
    jailbreak_method= "FigStep"

    eval_dataset = REBenchDataset(csv_path=csv_path,jailbreak_method =jailbreak_method,image_dir = images_path)

    batch_size = 8 # 只用于加载数据集，不涉及推理，这里的推理是单条推理
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn,
    ) # 此时的dataset 已经被分成batch_size个batch了

    # 初始化采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        repetition_penalty=repetition_penalty
    )

    # 初始化LLM
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=device_num,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 20},
        gpu_memory_utilization=0.9
    )
    result = {}
    counter = 0
    for batch in tqdm(eval_dataloader):
        # counter+=1
        # if counter == 2:
        #     break
        questions = batch["question"]
        images_path = batch["image"]
        task_ids = batch["task_id"]
        original_querys = batch["original_query"]
        resources = batch["from"]
        policies = batch["policy"]
        inputs = []
        for question,image in zip(questions, images_path):
            images = [PIL.Image.open(image)]
            try:
                question = "<image>\n" + question 
            except:
                question = "<image>\n" + " "

            prompt = f"<｜begin▁of▁sentence｜><｜User｜>\n{question}<｜Assistant｜><think>\n"
            buffer={
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": images
                    },
                    }
            inputs.append(buffer)

        try:
            outputs = llm.generate(inputs,sampling_params=sampling_params)
        except Exception as e:
            print(e)
            save_my_file(result)
            raise
        for i,o in enumerate(outputs):
            response = o.outputs[0].text
            task_id = task_ids[i]
            result[f"{task_id}"] = {
                "original_query": original_querys[i],
                "jailbreak_query": questions[i],
                "answer": response,
                "image_path": images_path[i],
                "from": resources[i],
                "policy": policies[i]
            }
    with open(f"test_resulst/{testing_model}_{jailbreak_method}.json", "w",encoding= 'utf-8') as f:
        json.dump(result,f,indent =4)

if __name__ == '__main__':
    main()