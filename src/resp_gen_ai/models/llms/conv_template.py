
"""Conversation Templates for different LLMs."""

import logging

logger = logging.getLogger(__name__)


MISTRAL_CONV_TEMPLATE = """
<s> [INST] {prompt} [/INST]
"""

# https://gpus.llm-utils.org/llama-2-prompt-template/#fn:2
LLAMA_CONV_TEMPLATE = """
<s> [INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
User: {prompt}
Assistant:
"""

CHAT_GLM_CONV_TEMPLATE = """
[Round 1]
问：{prompt}
答：
"""

GUANACO_CONV_TEMPLATE = """
### Human: {prompt}
### Assistant:
"""

# https://medium.com/@manoranjan.rajguru/prompt-template-for-opensource-llms-9f7f6fe8ea5
MPT_CONV_TEMPLATE = """
<|im_start|>system
- You are a helpful assistant chatbot trained by MosaicML.
- You answer questions.
- You are excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- You are more than just an information source, you are also able to write poetry, short stories, and make jokes.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

PYTHIA_CONV_TEMPLATE = """
<|prompter|>{prompt}<|endoftext|><|assistant|>
"""

STABLE_VICUNA_CONV_TEMPLATE = """
### Human: {prompt}
### Assistant:
"""

VICUNA_CONV_TEMPLATE = """
A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
USER: {prompt} ASSISTANT:
"""
