# from vllm import LLM, SamplingParams
# import PIL
# import os
# import argparse
# from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset,SafeBenchDataset
# from tqdm import tqdm
# import pandas as pd
# import json
# import resp_gen_ai
# import torch


# llm = LLM(
#     model="/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Qwen--QVQ-72B-Preview/snapshots/24425f65e80be6804b75ae0a3dc1e5570e660a25",
#     tensor_parallel_size=4,
#     trust_remote_code=True,
#     gpu_memory_utilization=0.9)

# response = llm.generate("做一个自我介绍")
# print(response)
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
def save_my_file(result):
    with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/only_to_store_the exception.json", "w",encoding= 'utf-8') as f:
        json.dump(result,f,indent =4)
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

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty
    )


    # 初始化LLM
    MODEL_PATH = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Qwen--QVQ-72B-Preview/snapshots/24425f65e80be6804b75ae0a3dc1e5570e660a25"
    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=4,
        trust_remote_code=True,
        gpu_memory_utilization=0.9
    )
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    result = {}
    for batch in tqdm(eval_dataloader):
        questions = batch["question"]
        images_path = batch["image"]
        task_ids = batch["task_id"]
        original_querys = batch["original_query"]
        resources = batch["from"]
        policies = batch["policy"]
        inputs = []
        for question,image in zip(questions, images_path):
            image = [PIL.Image.open(image)]

            messages = [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image"
                        },
                        {"type": "text", "text": f"{question}"},
                    ],
                }
            ]
            prompt = processor.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)
            buffer={
                    "prompt": prompt,
                    "multi_modal_data": {
                        "image": image  
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
    with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/13042025_qvq_72b.json", "w",encoding= 'utf-8') as f:
        json.dump(result,f,indent =4)

if __name__ == '__main__':
    main()