
"""ChatGLM."""

import logging

from resp_gen_ai.models import load_lm

logger = logging.getLogger(__name__)

# path can be either a local path to the model or a huggingface model name
# MODEL_PATH = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub/models--THUDM--chatglm2-6b/snapshots/7fabe56db91e085c9c027f56f1c654d137bdba40"
MODEL_PATH = "THUDM/chatglm2-6b"
model = load_lm(testing_model="ChatGLM2-6B", model_path=MODEL_PATH)
