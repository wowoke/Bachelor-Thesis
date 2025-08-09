import json
import re
import os

input_file = "multi_jailbreak_results/evaluation/"  # put your file path here, (not a folder path).
output_file = "multi_jailbreak_results/evaluation/final_VLM-R1.json"  # 输出文件

def extract_label_answer(entry):
    value = entry.get("same with label?", "").strip().lower()

    match = re.search(r'[\[\{]\s*(yes|no)\s*[\]\}]', value)
    if match:
        return match.group(1)
    
    if value in ("yes", "no"):
        return value

    if value.endswith("yes"):
        return "yes"
    if value.endswith("no"):
        return "no"

    return None

def normalize_same_with_label(data):
    for item in data:
        extracted = extract_label_answer(item)
        item["parsed_same_with_label"] = extracted
    return data

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

result = normalize_same_with_label(data)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

print(f"done {output_file}")