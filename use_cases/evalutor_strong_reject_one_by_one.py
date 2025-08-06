
from resp_gen_ai.metrics.jailbreak.evaluate import strongreject_finetuned
from tqdm import tqdm
import json
import os

folder_path = "REBench/results/basemodel_results/deepseek-R1-Distill-Qwen-32B"
json_files = ["REBench/results/baseline/baseline_Open_R1_multimodal.json"]

for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        full_path = os.path.join(folder_path, file_name)
        json_files.append(full_path)
json_files = ["REBench/results/baseline/baseline_Open_R1_multimodal.json"]
print("1")
for file in json_files:
    with open(file,"r",encoding ="utf-8") as f:
        data = json.load(f)
    judge_model_path = "qylu4156/strongreject-15k-v1"
    print("2")
    i =0
    buffer =0
    for key,dic in tqdm(data.items()):
        input_of_strong_reject = {
            "forbidden_prompt": [dic["jailbreak_query"]],
            "response": [dic["answer"]]
        }
        score = strongreject_finetuned(input_of_strong_reject, model_name = judge_model_path)
        score = score["score"][0]
        dic["score"] = score
        buffer += score
        i += 1
        if i == 5:
            break
    with open(f"test_results/strong_reject_{file_name}","w",encoding = 'utf-8') as f:
        json.dump(data,f,indent = 4)
    print(f"{file_name} json generated!!")


