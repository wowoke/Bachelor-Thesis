import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import LLMBaseModel


class LlamaGuard3_8B(LLMBaseModel):
    def __init__(self,model_path,model_type = "",dtype = torch.bfloat16,device= "cuda"):
        super().__init__(model_path,device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map=device)

    def generate(self,question,answer): # 接受的是两个string
        chat = [
            {"role":"user","content":f"{question}"},
            {"role":"assistant","content":f"{answer}"}
        ]
        input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.model.device)
        output = self.model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
        prompt_len = input_ids.shape[-1]
        return self.tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)









