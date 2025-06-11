import json
from resp_gen_ai.metrics.jailbreak.evaluate import calculate_asr_by_checking_refusal_words,strongreject_finetuned
from tqdm import tqdm

with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/question_and_answer_for_strong_reject.json",
        "r", encoding= 'utf-8') as f:
        data = json.load(f)
# data["response"]
# data["forbidden_prompt"]
import pandas as pd

df = pd.read_csv("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/VRP_query_specific/prompt/full_ours/vrp.csv")        # 读取 CSV
my_list = df["redteam_query"].tolist()     
data["forbidden_prompt"] = my_list
def split_dict_into_batches(data_dict, batch_size):
    total_len = len(next(iter(data_dict.values())))  # 假设每个 key 对应的 list 长度一样
    batches = []

    for i in range(0, total_len, batch_size):
        batch = {key: value[i:i + batch_size] for key, value in data_dict.items()}
        batches.append(batch)

    return batches

def merge_batches(batches):
    merged = {}
    for batch in batches:
        for key, value in batch.items():
            if key not in merged:
                merged[key] = []
            merged[key].extend(value)
    return merged


batches = split_dict_into_batches(data,8)
result = []

for batch in tqdm(batches):
        scores = strongreject_finetuned(batch, model_name = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--qylu4156--strongreject-15k-v1/snapshots/4bd893d32390d2cace4f067dc2e3ef5294fd78a2",)
        result.append(scores)
final_result = merge_batches(result)

with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/1111112.json",
        "w", encoding= 'utf-8') as f:
        json.dump(final_result,f,indent = 4)
print("Jawohl!")