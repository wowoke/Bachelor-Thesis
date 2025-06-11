
"""Mistral."""

import logging

# Load model directly
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLMBaseModel, PromptTemplate
from .conv_template import MISTRAL_CONV_TEMPLATE

logger = logging.getLogger(__name__)


class LLMMistral(LLMBaseModel):
    def __init__(self, model_path, model_type="7B-Instruct", device="cuda:0"):
        super().__init__(model_path, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.add_special_tokens(
            {
                "pad_token": self.tokenizer.eos_token,
            }
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map=device)
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.to(device)
        self.model.eval()
        assert model_type != "", "Please specify the model type"
        self.model_type = model_type
        self.prompt_template = PromptTemplate(MISTRAL_CONV_TEMPLATE)

    def generate(
        self,
        instruction: str,
        do_sample: bool = True,
        max_new_tokens: int = 300,
        num_beams: int = 1,
        min_length: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        length_penalty: float = 1,
        temperature: float = 1.0,
    ) -> str:
        """
        Generate a response from the model Mistral.
        :param instruction: the instruction to generate a response for
        :type instruction: str
        :param do_sample: whether to sample or not
        :type do_sample: bool
        :param max_new_tokens: the maximum number of tokens to generate
        :type max_new_tokens: int
        :param num_beams: the number of beams to use
        :type num_beams: int
        :param min_length: the minimum length of the generated text
        :type min_length: int
        :param top_p: the top p value
        :type top_p: int
        :param repetition_penalty: the repetition penalty
        :type repetition_penalty: float
        :param length_penalty: the length penalty
        :type length_penalty: float
        :param temperature: the temperature
        :type temperature: float
        :return: the generated response
        :rtype: str.
        """
        inputs = self.prompt_template(instruction)
        inputs = self.tokenizer(inputs, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)
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
        )
        output_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        output_text = output_text.split("[/INST]")[-1].strip()
        return output_text

    def __str__(self):
        return f"LLMMistral-{self.model_type}"
