import math
import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from .base import LLMBaseModel


safe_token = "No"
unsafe_token = "Yes"

def parse_output(output, input_len):
    label = None
    
    full_res = self.tokenizer.decode(output.sequences[:,input_len+1:][0],skip_special_tokens=True).strip()
    confidence_level = full_res.removeprefix('<confidence>').removesuffix('</confidence>').strip()
    res = self.tokenizer.decode(output.sequences[:,input_len:input_len+1][0],skip_special_tokens=True).strip()
    if unsafe_token.lower() == res.lower():
        label = unsafe_token
    elif safe_token.lower() == res.lower():
        label = safe_token
    else:
        label = "Failed"

    return label, confidence_level

class GraniteGuardian3_5b(LLMBaseModel):
    
    def __init__(self,model_path,model_type="",device ="cuda:0"):
        super().__init__(model_path,device)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def parse_output(self, output, input_len):
        label = None
        
        full_res = self.tokenizer.decode(output.sequences[:,input_len+1:][0],skip_special_tokens=True).strip()
        confidence_level = full_res.removeprefix('<confidence>').removesuffix('</confidence>').strip()
        res = self.tokenizer.decode(output.sequences[:,input_len:input_len+1][0],skip_special_tokens=True).strip()
        if unsafe_token.lower() == res.lower():
            label = unsafe_token
        elif safe_token.lower() == res.lower():
            label = safe_token
        else:
            label = "Failed"

        return label, confidence_level

    def generate(self,question,answer):

        messages = [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        # Please note that the default risk definition is of `harm`. If a config is not specified, this behavior will be applied.
        guardian_config = {"risk_name": "harm"}
        input_ids = self.tokenizer.apply_chat_template(
            messages, guardian_config = guardian_config, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        input_len = input_ids.shape[1]

        self.model.eval()

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                do_sample=False,
                max_new_tokens=20,
                return_dict_in_generate=True,
                output_scores=True,
            )
        return self.parse_output(output, input_len)
