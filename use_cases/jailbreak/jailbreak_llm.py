"""Jailbreak Meta Llama3-8B."""

import logging
import torch
import os
from resp_gen_ai.datasets.jailbreaks.jailbreak_datasets import JailbreakLMDataset
from resp_gen_ai.metrics.jailbreak.evaluate import calculate_asr_by_checking_refusal_words
from resp_gen_ai.models import load_lm
import resp_gen_ai
from tqdm import tqdm

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s:%(asctime)s:%(name)s:%(filename)s:%(lineno)d]\t %(message)s",
)

def custom_collate_fn(batch):
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [item[key] for item in batch]
    return collated_batch



def main():
    # Load Model
    model = load_lm(testing_model="Llama3-8B", model_path="meta-llama/Meta-Llama-3-8B")

    # Load Dataset
    base_dir = os.path.join(os.path.dirname(resp_gen_ai.__file__), "../../datasets/jailbreak_datasets/")
    data_path = os.path.join(base_dir, "harmful_behaviors_filtered-aim.json")
    print(data_path)
    eval_dataset = JailbreakLMDataset(harmful_behaviors_json=data_path)
    batch_size = 8
    eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=batch_size,
            sampler=None,
            shuffle=False,
            num_workers=8,
            collate_fn=custom_collate_fn,
        )
    responses = []
    for batch in tqdm(eval_dataloader):
        prompts = batch["instruction_with_behavior"]
        # pseudo batch mode here so far;
        for i in range(len(prompts)):
            inst = prompts[i]
            inst = eval_dataset.input_function(inst)
            logger.info(f"input: {inst}")
            resp = model.generate(
                inst,
            )
            resp = eval_dataset.output_function(resp)
            logger.info(f"response: {resp}")
            responses.append(resp)

        break

    asr = calculate_asr_by_checking_refusal_words(responses)
    logger.info(f"ASR: {asr}")


if __name__ == "__main__":
    main()
