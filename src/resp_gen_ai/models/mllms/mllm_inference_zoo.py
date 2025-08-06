import os
import sys

import yaml

import resp_gen_ai.models.mllms as mllms

sys.path.append(os.getcwd())


def setup_lm_weights_path(model_name, lm_path):
    """Set up the path to the language model weights in the model config file.
    Needed in MiniGPT4 models.


    Args:
        model_name (_type_): The model name.
        lm_path (_type_): LM path

    Raises:
        NotImplementedError: If the model name is not supported.
    """
    if model_name == "MiniGPT4-13B":
        lm_config_path = os.path.join(
            os.path.abspath(
                mllms.__file__,
            ).split(
                "__init__.py"
            )[0],
            "minigpt4_utils",
            "configs",
            "models",
            "minigpt4_13b.yaml",
        )
        lm_config = yaml.load(open(lm_config_path), Loader=yaml.FullLoader)
        if lm_config["model"]["llama_model"] == lm_path:
            return
        lm_config["model"]["llama_model"] = lm_path
        yaml.dump(lm_config, open(lm_config_path, "w"))
    else:
        raise NotImplementedError


def load_mllm(testing_model, model_path=None, lm_path=None):
    """Load the model from the model zoo.

    Args:
        testing_model (_type_): _description_
        model_path (_type_, optional): _description_. Defaults to None.
        lm_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # if testing_model == "Flamingo":
    #     from openflamingo_modeling import MLLMFlamingo
    #     ckpt_path = f'{cache_dir}/open_flamingo_9b_hf'
    #     model = MLLMFlamingo(ckpt_path, clip_vision_encoder_path="ViT-L-14", clip_vision_encoder_pretrained="openai", device="cuda:0")
    if testing_model == "MiniGPT4-7B":  ## mmqads
        from .minigpt4_modeling import MLLMMiniGPT4

        # model = MLLMMiniGPT4(f"{cache_dir}/minigpt4/minigpt4_eval_7b.yaml")
        assert (
            "7b" in model_path
        ), "Model path does not contain 7b, are your sure you are using the right model config?"
        model = MLLMMiniGPT4(model_path=model_path, model_type="minigpt4-7b")
    elif testing_model == "MiniGPT4v2":
        from .minigpt4_modeling import MLLMMiniGPT4

        # model = MLLMMiniGPT4(f"{cache_dir}/minigpt4/minigptv2_eval.yaml")
        assert (
            "v2" in model_path
        ), "Model path does not contain v2, are your sure you are using the right model config?"
        model = MLLMMiniGPT4(model_path=model_path, model_type="minigpt4-v2")
    elif testing_model == "MiniGPT4-llama2":  ## mmqads
        from .minigpt4_modeling import MLLMMiniGPT4

        assert (
            "llama2" in model_path
        ), "Model path does not contain llama2, are your sure you are using the right model config?"
        # model = MLLMMiniGPT4(f"{cache_dir}/minigpt4/minigpt4_eval_llama2.yaml")
        model = MLLMMiniGPT4(model_path=model_path, model_type="minigpt4-llama2")
    elif testing_model == "MiniGPT4-13B":  ## mmqads
        from .minigpt4_modeling import MLLMMiniGPT4

        if lm_path is not None:
            setup_lm_weights_path(testing_model, lm_path)
        assert (
            "13b" in model_path
        ), "Model path does not contain 13b, are your sure you are using the right model config?"
        # model = MLLMMiniGPT4(f"{cache_dir}/minigpt4/minigpt4_eval_13b.yaml")
        model = MLLMMiniGPT4(model_path=model_path, model_type="minigpt4-13b")
    elif testing_model == "LLaVA-7B":  ## mmqa
        from .llava_modeling import MLLMLLaVA

        assert (
            "llava-7b" in model_path
        ), "Model path does not contain llava-7b, are your sure you are using the right model config?"
        model = MLLMLLaVA(
            # {"model_path": f"{cache_dir}/llava/llava-7b", "device": 0, "do_sample": True}
            {"model_path": model_path, "device": 0, "do_sample": True},
            model_type="llava-7b",
        )
    elif testing_model == "LLaVA_llama2-13B":  ## mmqa
        from .llava_modeling import MLLMLLaVA

        assert (
            "llava-llama-2-13b" in model_path
        ), "Model path does not contain llava-llama-2-13b, are your sure you are using the right model config?"
        model = MLLMLLaVA(
            {
                # "model_path": f"{cache_dir}/llava/llava-llama-2-13b-chat",
                "model_path": model_path,
                "device": 0,
                "do_sample": True,
            },
            model_type="llava-llama2-13b",
        )
    elif testing_model == "LLaVA_llama2-7B-LoRA":  ## mmqa
        from .llava_modeling import MLLMLLaVA

        assert (
            "llava-llama-2-7b" in model_path
        ), "Model path does not contain llava-llama-2-7b, are your sure you are using the right model config?"
        model = MLLMLLaVA(
            {
                # "model_path": f"{cache_dir}/llava/llava-llama-2-13b-chat",
                "model_path": model_path,
                "device": 0,
                "do_sample": True,
            },
            model_type="llava-llama2-7b",
        )

    elif testing_model == "LLaVAv1.5-7B":
        from .llava_modeling import MLLMLLaVA

        assert (
            "llava-v1.5-7b" in model_path
        ), "Model path does not contain llava-v1.5-7b, are your sure you are using the right model config?"
        model = MLLMLLaVA(
            # {"model_path": f"{cache_dir}/llava/llava-v1.5-7b", "device": 0, "do_sample": False}
            {"model_path": model_path, "device": 0, "do_sample": False},
            model_type="llava-v1.5-7b",
        )
    elif testing_model == "LLaVAv1.5-13B":
        from .llava_modeling import MLLMLLaVA

        assert (
            "llava-v1.5-13b" in model_path
        ), "Model path does not contain llava-v1.5-13b, are your sure you are using the right model config?"
        model = MLLMLLaVA(
            # {"model_path": f"{cache_dir}/llava/llava-v1.5-13b", "device": 0, "do_sample": False}
            {"model_path": model_path, "device": 0, "do_sample": False},
            model_type="llava-v1.5-13b",
        )

    elif testing_model == "LlamaAdapterV2":  ## mmqa
        from .llama_adapter_v2_modeling import MLLMLlamaAdapterV2

        model = MLLMLlamaAdapterV2(
            # model_path=f"{cache_dir}/llama-adapterv2/BIAS-7B.pth",
            model_path=model_path,
            device="cuda:0",
            # base_lm_path=f"{cache_dir}/LLaMA-7B",
            base_lm_path=lm_path,
        )
    # elif testing_model == "mPLUGOwl": ## flamingo
    #     from mplug_owl_modeling import  MLLMmPLUGOwl
    #     model = MLLMmPLUGOwl(f"{cache_dir}/mplug-owl-7b-ft")
    elif testing_model == "mPLUGOwl2":
        from .mplug_owl2_modeling import MLLMmPLUGOwl2

        model = MLLMmPLUGOwl2(model_path)

    elif testing_model == "InstructBLIP-13B":  ## mmqa
        from .instruct_blip_modeling import MLLMInstructBLIP2

        model = MLLMInstructBLIP2(model_path, model_type="13B")
    elif testing_model == "InstructBLIP-7B":
        from .instruct_blip_modeling import MLLMInstructBLIP2

        model = MLLMInstructBLIP2(model_path, model_type="7B")

        # model = MLLMInstructBLIP2(f"{cache_dir}/instructblip-vicuna-7b")
    # elif testing_model == "InstructBLIP2-13B": ## mmqa
    #     from .instruct_blip_modeling import MLLMInstructBLIP2
    #     model = MLLMInstructBLIP2(f"{cache_dir}/instructblip-vicuna-13b")
    # elif testing_model == "InstructBLIP2-FlanT5-xl": ## mmqa
    #     from .instruct_blip_modeling import MLLMInstructBLIP2
    #     model = MLLMInstructBLIP2(f"{cache_dir}/instructblip-flan-t5-xl")
    # elif testing_model == "InstructBLIP2-FlanT5-xxl": ## mmqa
    #     from .instruct_blip_modeling import MLLMInstructBLIP2
    #     model = MLLMInstructBLIP2(f"{cache_dir}/instructblip-flan-t5-xxl")
    # #
    elif testing_model == "Qwen-VL-Chat":  ## mmqa
        from .qwen_vl_modeling import MLLMQwenVL

        model = MLLMQwenVL(model_path)
    elif testing_model == "CogVLM":
        from .cogvlm_modeling import MLLMCogVLMChat

        model = MLLMCogVLMChat(
            model_path=model_path,
        )

    elif testing_model == "Fuyu":  ## mmqa
        from .fuyu_modeling import MLLMFuyu

        model = MLLMFuyu(model_path=model_path)
    elif testing_model == "InternVL2.5":
        from .internvl2_5_modeling import MLLMInternVL
        
        model = MLLMInternVL(model_path=model_path)
    elif testing_model == "Qwen2-VL-Instruct":
        from .qwen2_vl_Instruct_modeling import MLLMQwen2VLInstruct
        
        model = MLLMQwen2VLInstruct(model_path=model_path)

    elif testing_model == "PKU-DS-V":
        from .pku_ds_v_modeling import RMLLMPkuDsV

        model = RMLLMPkuDsV(model_path = model_path)

    elif testing_model == "Qwen2-VL":
        from .qwen2_vl_modeling import MLLMQwen2VL

        model = MLLMQwen2VL(model_path = model_path)
    elif testing_model == "Qwen2-VL-vllm":
        from .qwen2_vl_instruct_vllm_modeling import MLLMQwen2VLInstructVLLM

        model = MLLMQwen2VLInstructVLLM(model_path=model_path)
    return model
