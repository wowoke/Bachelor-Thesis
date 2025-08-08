import logging
import pandas as pd
import json
from resp_gen_ai.models import load_rmllm,load_mllm
from tqdm import tqdm
import os


csv_path = "multimodalpragmatic_zero_shot.csv" # Put the csv path here
df = pd.read_csv(csv_path)


image_paths = df["image_path"].tolist()
labels = df["label"].tolist()
instructions = df["textprompt"].tolist()

# Choose the model you want to run

# testing_model = "QVQ-72B"
# model_path ="Qwen/QVQ-72B-Preview"

# testing_model = "PKU-DS-V"
# model_path = "PKU-Alignment/Align-DS-V"

# testing_model = "Skywork-R1V-vllm"
# model_path = "Skywork/Skywork-R1V-38B"

# testing_model = "Visual_RFT"
# model_path = "Zery/Qwen2-VL-7B_visual_rft_lisa_IoU_reward"

# testing_model = "VLM-R1"
# model_path ="omlab/VLM-R1-Qwen2.5VL-3B-OVD-0321"

# testing_model = "Open_R1_multimodal"
# model_path = "lmms-lab/Qwen2-VL-7B-GRPO-8k"


# testing_model = "Qwen2-VL-vllm"
# size = "7B"
# model_path =  "Qwen/Qwen2-VL-7B-Instruct"

# testing_model = "Qwen2-VL-vllm"
# size = "72B"
# model_path =  "Qwen/Qwen2-VL-72B"

testing_model = "Qwen2-VL-vllm"
size = "2.5_3B"
model_path =  "Qwen/Qwen2.5-VL-3B-Instruct"


llm = load_mllm(testing_model=testing_model, model_path=model_path)


batch_size = 50 


output_data = []


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

os.makedirs("pragmatic_jailbreak_results", exist_ok=True)
with open(f"pragmatic_jailbreak_results/{testing_model}_{size}.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print("✅")