
"""Base class wrapper for MLLMs."""

import logging

import torch

logger = logging.getLogger(__name__)


class MLLMForJailbreak(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.model_path = model_path

    def get_loss(self, image, text_prompt, target_text):
        """
        images: (list) a list of images( PIL.Image.open(img).convert('RGB') )
        text_prompt: (list) a list of prompts
        target text: (list) a list of target texts
        Return: (tensor) loss.
        """
        raise NotImplementedError


class MLLMBaseModel(torch.nn.Module):
    def __init__(self, model_path, device):
        super().__init__()
        self.device = device
        self.model_path = model_path

    def forward(self, instruction, images):
        return self.generate(instruction, images)

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response.
        """
        raise NotImplementedError

    def __str__(self):
        raise NotImplementedError
