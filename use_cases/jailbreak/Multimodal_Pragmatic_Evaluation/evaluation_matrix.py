import json




file = "Visual_RFT.json" # Put your file path here

with open(file,"r",encoding="utf-8") as f:
    data = json.load(f)
TP = 0
TN = 0
FP = 0
FN = 0
for dic in data:
    label = dic["label"]
    result = dic["parsed_same_with_label"]
    if label == "safe":
        if result == "yes":
            TP +=1
        elif result == "no":
            FP +=1
        else:
            pass
    elif label == "unsafe":
        if result == "yes":
            TN +=1
        elif result == "no":
            FN +=1
        else:
            pass
dic = {"TP":TP,
       "FP":FP,
       "TN":TN,
       "FN":FN}
data.append(dic)
with open(file,"w",encoding="utf-8")as f:
    json.dump(data,f,indent = 4)
print(f"{file} ","done")
