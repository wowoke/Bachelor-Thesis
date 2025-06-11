from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from .base import LLMBaseModel


class LlamaGuard3_1B(LLMBaseModel):

    def __init__(self,model_path,device= "cuda:0"):
        super().__init__(model_path,device)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def generate(self,question,answer):

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text", 
                        "text": f"{question}"
                    },
                ],
            },
            {
                "role":"assistant",
                "content":[
                    {
                        "type":"text",
                        "text":f"{answer}"
                    }
                ]
            }
        ]

        input_ids = self.tokenizer.apply_chat_template(
            conversation, return_tensors="pt"
        ).to(self.model.device)

        prompt_len = input_ids.shape[1]
        output = self.model.generate(
            input_ids,
            max_new_tokens=20,
            pad_token_id=0,
            do_sample=False,  
            top_p = None,
            temperature  = None
        )
        generated_tokens = output[:, prompt_len:]
        return self.tokenizer.decode(generated_tokens[0])

