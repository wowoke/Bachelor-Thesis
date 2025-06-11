import os
import sys

import yaml

import resp_gen_ai.models.rmllms as rmllms

sys.path.append(os.getcwd())


def load_rmllm(testing_model, model_path=None, lm_path=None):
    """Load the model from the model zoo.

    Args:
        testing_model (_type_): _description_
        model_path (_type_, optional): _description_. Defaults to None.
        lm_path (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    if testing_model == "PKU-DS-V":
        from .pku_ds_v_modeling import RMLLMPkuDsV

        model = RMLLMPkuDsV(model_path = model_path)
    elif testing_model =="Eureka_32b":
        from .eureka_32b_modeling import Eureka_32b

        model = Eureka_32b(model_path = model_path)
    elif testing_model == "Skywork-R1V":
        from .skywork_R1V_modeling import Skywork_R1V

        model = Skywork_R1V(model_path= model_path)
    elif testing_model == "Visual_RFT":
        from .visual_rft_modeling import Visual_RFT

        model = Visual_RFT(model_path = model_path)
    elif testing_model == "Open_R1_multimodal":
        from .open_R1_multimodal_modeling import open_r1_multimodal

        model = open_r1_multimodal(model_path = model_path)
    return model
