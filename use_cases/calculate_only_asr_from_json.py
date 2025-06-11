import json
from tqdm import tqdm

with open('evaluation_DS_V.json','r') as f:
    data = json.load(f)
pos = 0
for dic in tqdm(data):
    if dic['label'] == "Yes":
        pos +=1

print(pos/len(data))
    