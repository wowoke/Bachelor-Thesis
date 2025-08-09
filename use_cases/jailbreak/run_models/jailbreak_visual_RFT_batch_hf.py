import logging
import torch
import os

os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_IB_TIMEOUT"] = "7200"

from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset,REBenchDataset
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






def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch



def main():
    # load models
    batch_size = 14 
    accelerator = Accelerator()
    testing_model = "Visual_RFT"
    model_path = "Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward"


    csv_path = "REBench_Dataset/Baseline/REBench_Baseline.csv" # Dataset Path
    images_path = "REBench_Dataset/image.png"
    jailbreak_method= "baseline"


    print("loading model")
    model = load_rmllm(testing_model=testing_model, model_path=model_path)
    print("model loaded")
    # load dataset
    print("loading dataset")
    eval_dataset = REBenchDataset(csv_path=csv_path,jailbreak_method = jailbreak_method,image_dir = images_path )
    
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=custom_collate_fn,
    ) 
    eval_dataloader = accelerator.prepare(eval_dataloader)
    print("dataset loaded")
    local_result = {}
    answer = []
    for batch in tqdm(eval_dataloader):
        questions = batch["question"]
        images = batch["image"]
        task_ids = batch["task_id"]
        original_query = batch["original_query"]
        resource = batch["from"]
        policy = batch["policy"]
        resp = model.generate_batch(
            questions,
            images, 
        )
        for i in range(len(questions)):
            id = task_ids[i]
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
        with open(f"test_results/{testing_model}_{jailbreak_method}.json", "w",encoding= 'utf-8') as f:
            json.dump(merged_result,f,indent =4)
        print("json generated!")
    accelerator.end_training()


if __name__ == "__main__":
    import torch.distributed as dist
    import datetime
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized() is False:
            dist.init_process_group(
                backend="nccl",
                timeout=datetime.timedelta(seconds=7200)  # 设置为两小时
            )
    except:
        pass

    main()