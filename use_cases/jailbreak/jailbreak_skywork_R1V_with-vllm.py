from vllm import LLM, SamplingParams
import PIL
import os
import argparse
from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset,SafeBenchDataset
from tqdm import tqdm
import pandas as pd
import json
import resp_gen_ai
import torch

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Run inference with Skywork-R1V model using vLLM.")
    # Model configuration
    parser.add_argument('--model_path', type=str, default='Skywork/Skywork-R1V-38B', 
                       help="Path to the model.")
    parser.add_argument('--tensor_parallel_size', type=int, default=4, 
                       help="Number of GPUs for tensor parallelism.")
    
    # Input parameters
    # parser.add_argument('--image_paths', type=str, nargs='+', required=True, 
    #                    help="Path(s) to the image(s).")
    # parser.add_argument('--question', type=str, required=True, 
    #                    help="Question to ask the model.")
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=0,
                       help="Temperature for sampling (higher = more creative).")
    parser.add_argument('--max_tokens', type=int, default=64000,
                       help="Maximum number of tokens to generate.")
    parser.add_argument('--repetition_penalty', type=float, default=1.05,
                       help="Penalty for repeated tokens (1.0 = no penalty).")
    parser.add_argument('--top_p', type=float, default=0.95,
                       help="Top-p (nucleus) sampling probability.")
    
    args = parser.parse_args()

    csv_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/VRP_query_specific/prompt/full_ours/vrp.csv"
    image_dir = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/VRP_query_specific/image/ours_full"

    # 加载数据集
    eval_dataset = SafeBenchDataset(csv_path=csv_path, image_dir=image_dir,jailbreak_method ="VRP")

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
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty
    )

    # 初始化LLM
    llm = LLM(
        model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Skywork--Skywork-R1V-38B/snapshots/c00c3791112a62823098f0721f5a1065314f7076",
        tensor_parallel_size=2,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 20},
        gpu_memory_utilization=0.9
    )
    result = {}
    for batch in tqdm(eval_dataloader):
        questions = batch["question"]
        images_path = batch["image"]
        task_ids = batch["task_id"]
        original_querys = batch["original_query"]
        resources = batch["from"]
        policies = batch["policy"]
        for i in range(len(questions)):
            question = questions[i]
            image = images_path[i]
            task_id = task_ids[i]
            original_query = original_querys[i]
            resource = resources[i]
            policy = policies[i]
            if type(image) != list:
                image = [image]
            # 准备图像输入
            images = [PIL.Image.open(img_path) for img_path in image]
            # 构建prompt - 为每张图片添加<image>标记
            question = "<image>\n" * len(images) + question 
            prompt = f"<｜begin▁of▁sentence｜><｜User｜>\n{question}<｜Assistant｜><think>\n"
            
            # 执行推理 - 支持多图输入
            outputs = llm.generate(
                {
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": images  # 传入所有图片的列表
                    },
                },
                sampling_params=sampling_params
            )

            # 获取生成结果
            response = outputs[0].outputs[0].text
            result[f"{task_id}"] = {
                "original_query": original_query,
                "jailbreak_query": question,
                "answer": response,
                "image_path": image[0],
                "from": resource,
                "policy": policy
            }
    with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/skywork-09042025.json", "w",encoding= 'utf-8') as f:
        json.dump(result,f,indent =4)

if __name__ == '__main__':
    main()