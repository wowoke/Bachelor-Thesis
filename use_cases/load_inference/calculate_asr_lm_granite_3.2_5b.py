from resp_gen_ai.models import load_lm
import json
from tqdm import tqdm

MODEL_PATH = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--ibm-granite--granite-guardian-3.2-5b/snapshots/43e35985198a4cc97db3d489455aa96d1dd8a80e"
model = load_lm(testing_model="granite_guardian_3.2_5b",model_path=MODEL_PATH)

with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/output.json","r") as f:
    list_of_dict = json.load(f)


result = {"target_model": "PKU-DS-V",
          "Judge": "granite_3.2_5b"}
pos = 0
index = 0
for dic in tqdm(list_of_dict):

    question = dic['question']
    answer = dic['answer']

    label, confidence = model.generate(question,answer)

    if label == "Yes": # Yes means it is harmful
        pos +=1
    dic["label"] = label

    dic["confidence"] = confidence
    index += 1
    result[f"{index}"] = dic


asr = pos/len(list_of_dict)
print(asr)
result["asr"] = asr
with open("evaluation_DS_V.json",'w') as f:
    json.dump(result,f,indent = 4)

print("json文件已生成")







