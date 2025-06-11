
from resp_gen_ai.metrics.jailbreak.evaluate import strongreject_finetuned
from tqdm import tqdm
import json
import os

folder_path = '/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/VRP-results'
json_files = []

for file_name in os.listdir(folder_path):
    if file_name.endswith('.json'):
        full_path = os.path.join(folder_path, file_name)
        json_files.append(full_path)
for file in json_files:
    if file == "/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/VRP-results/pku-dsv-vrp-strong-reject.json":
        continue
    with open(file,"r",encoding ="utf-8") as f:
        data = json.load(f)
    judge_model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--qylu4156--strongreject-15k-v1/snapshots/4bd893d32390d2cace4f067dc2e3ef5294fd78a2"
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
        # if i == 5:
        #     break
    average = buffer/3952
    data["storng_reject"] = average
    print(average)
    if "open_r1" in file:
        model_name = "Open_R1"
    elif "pku-dsv" in file:
        model_name = "wrong"
    elif "qvq_72b" in file:
        model_name = "QVQ72B"
    elif "skywork" in file:
        model_name = "SkyworkR1V"
    elif "vlm-r1" in file:
        model_name = "VLM-R1"
    elif "visual_rft" in file:
        model_name = "visual_rft"
    with open(f"/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/VRP-results/with_evaluation/strong_reject_{model_name}.json","w",encoding = 'utf-8') as f:
        json.dump(data,f,indent = 4)
    print(f"{model_name}json generated!!")
    model_name = None