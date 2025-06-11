import logging
import torch
import os

os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_IB_TIMEOUT"] = "7200"

from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset,SafeBenchDataset
from resp_gen_ai.metrics.jailbreak.evaluate import calculate_asr_by_checking_refusal_words,strongreject_finetuned
from resp_gen_ai.models import load_rmllm, load_lm
import resp_gen_ai
from tqdm import tqdm
import pandas as pd
from PIL import Image
import json
from accelerate import Accelerator
import datetime

logger = logging.getLogger(__name__)


def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch



def main():
    # load models
    accelerator = Accelerator()
    testing_model = "Open_R1_multimodal"
    model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--lmms-lab--Qwen2-VL-7B-GRPO-8k/snapshots/a7f7d7e0b2558135072e26d8edcb188e9488fc0c"
    judge_model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--qylu4156--strongreject-15k-v1/snapshots/4bd893d32390d2cace4f067dc2e3ef5294fd78a2"
    judge_model = "qylu4156/strongreject-15k-v1"


    csv_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/cognitive_overload/veiled_attack/veiled_attack.csv"
    jailbreak_method= "cognitive_overload_veiled_attack"


    print("loading model")
    model = load_rmllm(testing_model=testing_model, model_path=model_path)
    print("model loaded")
    # load dataset
    print("loading dataset")
    eval_dataset = SafeBenchDataset(csv_path=csv_path,jailbreak_method = jailbreak_method)
    batch_size = 14 # 表示每次放进去8组数据，在推理过程中，这8组数据完全独立，互不影响。在训练过程中，这八组数据共同影响一次梯度的计算，对模型进行更新。

    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=custom_collate_fn,
    ) # 此时的dataset 已经被分成batch_size个batch了
    eval_dataloader = accelerator.prepare(eval_dataloader)
    print("dataset loaded")
    local_result = {}
    answer = []
    counter = 0
    for batch in tqdm(eval_dataloader): 
        # counter +=1
        # if counter ==2:
        #     break
        questions = batch["question"]
        images = batch["image"]
        task_ids = batch["task_id"]
        original_query = batch["original_query"]
        resource = batch["from"]
        policy = batch["policy"]
        resp = model.generate(
            questions,
            images
        )
        # print(type(questions))
        # print(type(resp))
        # input_of_strongreject_1 = {
        #     "forbidden_prompt": questions,
        #     "response": resp
        # }
        # input_of_strongreject_2 = {
        #     "forbidden_prompt": original_query,
        #     "response": resp          
        # }
        # print(type(input_of_strongreject_1))
        # scores_with_jailbreak_query = strongreject_finetuned(input_of_strongreject_1,model_name = judge_model_path)
        # scores_with_original_query = strongreject_finetuned(input_of_strongreject_2,model_name = judge_model_path)
        for i in range(batch_size):
            print(i)
            id = task_ids[i]
            print(f"id{id}")
            local_result[f"{id}"] = {
                "original_query": original_query[i],
                "jailbreak_query": questions[i],
                "answer": resp[i],
                "image_path": images[i],
                "from": resource[i],
                "policy": policy[i]
            }
    local_result = [local_result]
    accelerator.wait_for_everyone()
    all_results = accelerator.gather_for_metrics(local_result)
    if accelerator.is_main_process:
        merged_result = {}
        for d in all_results:
            for k, v in d.items():
                merged_result[k] = v
        with open(f"/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/{jailbreak_method}/{jailbreak_method}_{testing_model}.json", "w",encoding= 'utf-8') as f:
            json.dump(merged_result,f,indent =4)
        print("json generated!")
    accelerator.end_training()


if __name__ == "__main__":
    import torch.distributed as dist
    import datetime

    if torch.distributed.is_available() and torch.distributed.is_initialized() is False:
        dist.init_process_group(
            backend="nccl",
            timeout=datetime.timedelta(seconds=14400)  # 设置为两小时
        )
    main()