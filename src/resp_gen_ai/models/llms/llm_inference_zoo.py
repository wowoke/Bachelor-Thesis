
"""Entrance to Load LLM."""

import logging

logger = logging.getLogger(__name__)


def load_lm(testing_model, model_path=None, device="cuda:0"):
    if testing_model == "Llama2-7B":
        from .llama import LLMLlama

        model = LLMLlama(model_path, model_type="Llama2-7B", device=device)
    elif testing_model == "Llama3-8B":
        from .llama import LLMLlama

        model = LLMLlama(model_path, model_type="Llama3-8B", device=device)
    elif testing_model == "Llama-Guard":
        from .llama_guard import LLMLlamaGuard

        model = LLMLlamaGuard(model_path, device="cuda:0")
    elif testing_model == "Llama-Guard-4bit":
        from .llama_guard import LLMLlamaGuard

        model = LLMLlamaGuard(model_path, device="cuda:0", model_type="4bit")
    elif testing_model == "Vicuna-13B":
        from .vicuna import LLMVicuna

        model = LLMVicuna(model_path, model_type="Vicuna-13B", device=device)
    elif testing_model == "Vicuna-7B":
        from .vicuna import LLMVicuna

        model = LLMVicuna(model_path, model_type="Vicuna-7B", device=device)
    elif testing_model == "MPT-7B":
        from .mpt import LLMMPT

        model = LLMMPT(model_path, model_type="MPT-7B", device=device)
    elif testing_model == "Guanaco-13B":
        from .guanaco import LLMGuanaco

        model = LLMGuanaco(model_path, model_type="Guanaco-13B", device=device)
    elif testing_model == "Guanaco-7B":
        from .guanaco import LLMGuanaco

        model = LLMGuanaco(model_path, model_type="Guanaco-7B", device=device)
    elif testing_model == "Mistral-7B-Instruct":
        from .mistral import LLMMistral

        model = LLMMistral(model_path, model_type="7B-Instruct", device=device)
    elif testing_model == "Mistral-8x7B":
        from .mistral import LLMMistral

        model = LLMMistral(model_path, model_type="8x7B", device=device)
    elif testing_model == "ChatGLM2-6B":
        from .chatglm import LLMChatGLM

        model = LLMChatGLM(model_path, model_type="ChatGLM2-6B", device=device)
    elif testing_model == "Pythia-12B-Epoch3.5":
        from .pythia import LLMPythia

        model = LLMPythia(model_path, model_type="12B-Epoch3.5", device=device)

    elif testing_model == "StableVicuna-13B":
        from .stable_vicuna import LLMStableVicuna

        model = LLMStableVicuna(model_path, model_type="13B", device=device)
    elif testing_model == "Llama-Guard-3-1B":
        from .Llama_Guard_3_1B import LlamaGuard3_1B

        model = LlamaGuard3_1B(model_path,model_type = "llamaguard3-1b",device = device)
    elif testing_model == "granite_guardian_3.2_5b":
        from .granite_guardian_3_5b import GraniteGuardian3_5b

        model = GraniteGuardian3_5b(model_path,model_type = "granite-guardian3.2-5b",device = device)
    elif testing_model == "Llama-Guard-3-8B":
        from .llama_guard_3_8B import LlamaGuard3_8B

        model = LlamaGuard3_8B(model_path = model_path,device = device)
    else:
        raise NotImplementedError
    return model
