import json
with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/evaluation_results/1604FigStep.json","r", encoding = "utf-8") as f:
    data = json.load(f)
with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/FigStep/16042025_seond_round_figstep.json","r",encoding = 'utf-8') as f:
    data2 = json.load(f)
for ID,item in data.items():
    if item["figstep"]==None:
        for ID2,item2 in data2.items():
            if ID2 == ID:
                item["figstep"] = item2["second_figstep"]

with open("/dss/dssfs05/pn39qo/pn39qo-dss-0001/di93zun2/zhongyi/RespGenAI-zhongyi/use_cases/jailbreak/FigStep/1704_second_full_figsteps.json","w",encoding = 'utf-8') as f:
    json.dump(data,f,indent = 4)
print(len(data))