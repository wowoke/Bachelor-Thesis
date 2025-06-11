
"""LLaMA."""

import logging
import pdb

# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLMBaseModel, PromptTemplate, StoppingCriteriaList, StopOnTokens
from .conv_template import LLAMA_CONV_TEMPLATE

logger = logging.getLogger(__name__)


class LLMLlama(LLMBaseModel):
    def __init__(self, model_path, model_type="", device="cuda:0"):
        super().__init__(model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(device)
        self.model.eval()
        assert model_type != "", "Please specify the model type"
        self.model_type = model_type
        self.prompt_template = PromptTemplate(LLAMA_CONV_TEMPLATE)

    def generate(
        self,
        instruction,
        do_sample=True,
        max_new_tokens=300,
        num_beams=1,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.05,
        length_penalty=1,
        temperature=1.0,
        num_return_sequences=1,
        stop=None
    ):
        """
        instruction: (str) a string of instruction
        Return: (str) a string of generated response.
        """
        inputs = self.prompt_template(instruction)
        inputs = self.tokenizer(inputs, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        # Build stopping criteria only if a stop string is provided
        if stop:
            stop_ids = self.tokenizer(stop, add_special_tokens=False).input_ids
            stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_ids)])
        else:
            stopping_criteria = StoppingCriteriaList()  # empty list => no custom stop

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            stopping_criteria=stopping_criteria
        )
        if output.shape[0] == 1:
            output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            output_text = output_text.split("Assistant:")[-1].strip()
        else:
            output_text = [self.tokenizer.decode(o, skip_special_tokens=True) for o in output]
            output_text = [o.split("Assistant:")[-1].strip() for o in output_text]
        return output_text

    def __str__(self):
        return f"LLMLlama-{self.model_type}"
