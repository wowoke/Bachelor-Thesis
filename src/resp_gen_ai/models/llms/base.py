
"""Base LLM Model."""

import logging

import torch
from transformers import StoppingCriteria, StoppingCriteriaList

logger = logging.getLogger(__name__)

class StopOnTokens(StoppingCriteria):
    """
    Stop when a certain token sequence appears in the output IDs.
    """
    def __init__(self, stop_ids: list):
        super().__init__()
        self.stop_ids = stop_ids  # e.g. [1234, 333, 777] token IDs

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # We compare the last len(stop_ids) tokens in `input_ids[0]` to `self.stop_ids`
        if len(input_ids[0]) < len(self.stop_ids):
            return False
        if torch.equal(input_ids[0][-len(self.stop_ids):], torch.tensor(self.stop_ids, device=input_ids.device)):
            return True
        return False

class PromptTemplate:
    def __init__(self, template):
        self.template = template

    def __call__(self, prompt):
        return self.template.replace("{prompt}", prompt)


class LLMBaseModel(torch.nn.Module):
    """ """

    def __init__(self, model_path: str, device: [torch.device, str]):
        """Constructor for LLMBaseModel.
        :param model_path: the path to the model
        :type model_path: str
        :param device: the device to run the model on
        :type device: torch.device or str.
        """
        super().__init__()
        self.device = device
        self.model_path = model_path

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
    ):
        raise NotImplementedError

    def forward(self, instruction):
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
