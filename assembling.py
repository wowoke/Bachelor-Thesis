
#/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/.cache/huggingface/hub/llava-llama-2-7b-full
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel
from transformers import CLIPVisionModel

MODEL_PATH = "/dss/dssmcmlfs01/pn39qo/pn39qo-dss-0000/.cache/huggingface/hub/llava-llama-2-7b-full"

# **加载 Base Model**
base_model = AutoModelForCausalLM.from_pretrained(
    f"{MODEL_PATH}/base_model",
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True
)

# **加载 Vision Model**
vision_model = CLIPVisionModel.from_pretrained(f"{MODEL_PATH}/vision_model", local_files_only=True)

# **加载 LoRA 适配参数**
base_model = PeftModel.from_pretrained(base_model, f"{MODEL_PATH}/lora_adapter", local_files_only=True)

# **加载 Tokenizer**
tokenizer = AutoTokenizer.from_pretrained(f"{MODEL_PATH}/base_model", local_files_only=True)

# **测试模型**
input_text = "Describe the image in detail."
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
output = base_model.generate(**inputs, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
