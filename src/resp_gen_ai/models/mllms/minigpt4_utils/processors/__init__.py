"""
Copyright (c) 2022, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE_Lavis file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from resp_gen_ai.models.mllms.minigpt4_utils.common.registry import registry
from resp_gen_ai.models.mllms.minigpt4_utils.processors.base_processor import BaseProcessor
from resp_gen_ai.models.mllms.minigpt4_utils.processors.blip_processors import (
    Blip2ImageEvalProcessor,
    Blip2ImageTrainProcessor,
    BlipCaptionProcessor,
)

__all__ = [
    "BaseProcessor",
    "Blip2ImageEvalProcessor",
    "Blip2ImageTrainProcessor",
    "BlipCaptionProcessor",
]


def load_processor(name, cfg=None):
    """
    Example
    >>> processor = load_processor("alpro_video_train", cfg=None)
    """
    processor = registry.get_processor_class(name).from_config(cfg)

    return processor
