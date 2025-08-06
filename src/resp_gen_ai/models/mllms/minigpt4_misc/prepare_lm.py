
"""TODO."""

import logging

logger = logging.getLogger(__name__)

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Vision-CAIR/vicuna-7b",
    cache_dir="/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/.cache/huggingface/hub",
)
