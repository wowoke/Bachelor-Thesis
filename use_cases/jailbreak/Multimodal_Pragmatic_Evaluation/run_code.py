import logging
import pandas as pd
import json
from resp_gen_ai.models import load_rmllm,load_mllm
from tqdm import tqdm

# Step 1: 读取 CSV
csv_path = "multimodalpragmatic_zero_shot.csv"
df = pd.read_csv(csv_path)

# Step 2: 读取 image paths 和 text prompts（确保列名正确）
image_paths = df["image_path"].tolist()
labels = df["label"].tolist()
instructions = df["textprompt"].tolist()

# Step 3: 加载模型
# testing_model = "QVQ-72B"
# model_path ="/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Qwen--QVQ-72B-Preview/snapshots/24425f65e80be6804b75ae0a3dc1e5570e660a25"

# testing_model = "PKU-DS-V"
# model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--PKU-Alignment--Align-DS-V/snapshots/0a5c659e4c9fa78a6e64d2144d081725534599cb"

# testing_model = "Skywork-R1V-vllm"
# model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Skywork--Skywork-R1V-38B/snapshots/c00c3791112a62823098f0721f5a1065314f7076"

# testing_model = "Visual_RFT"
# model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Zery--Qwen2-VL-7B_visual_rft_lisa_IoU_reward/snapshots/cc595b1208351cf1b9ac53fd7245055dd935580e"

# testing_model = "VLM-R1"
# model_path ="/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--omlab--VLM-R1-Qwen2.5VL-3B-OVD-0321/snapshots/f2e8476efe7996d6c25d82507774e903f9a8471a"

# testing_model = "Open_R1_multimodal"
# model_path = "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--lmms-lab--Qwen2-VL-7B-GRPO-8k/snapshots/a7f7d7e0b2558135072e26d8edcb188e9488fc0c"


# testing_model = "Qwen2-VL-vllm"
# size = "7B"
# model_path =  "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/eed13092ef92e448dd6875b2a00151bd3f7db0ac"

# testing_model = "Qwen2-VL-vllm"
# size = "72B"
# model_path =  "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Qwen--Qwen2-VL-72B/snapshots/123bc2f135136461561679f1c9510bd57f886c8c"

testing_model = "Qwen2-VL-vllm"
size = "2.5_3B"
model_path =  "/dss/dssfs05/pn39qo/pn39qo-dss-0001/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3"


llm = load_mllm(testing_model=testing_model, model_path=model_path)

# 设置批量大小
batch_size = 50  # 可根据显存情况调整

# 保存所有结果
output_data = []

# 分批处理
count = 0
for i in tqdm(range(0, len(instructions), batch_size)):
    # count +=1
    # if count == 2:
    #     break
    batch_instructions = instructions[i:i + batch_size]
    batch_images = image_paths[i:i + batch_size]
    batch_label = labels[i:i+batch_size]


    batch_answers = llm.generate(
        instructions=batch_instructions,
        images=batch_images,
    )


    for image_path, instruction, answer,label in zip(batch_images, batch_instructions, batch_answers,batch_label):
        output_data.append({
            "image_path": image_path,
            "label": label,
            "answer": answer
            
        })

# Step 6: 保存为JSON文件
with open(f"multi_jailbreak_results/{testing_model}_{size}.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("✅")