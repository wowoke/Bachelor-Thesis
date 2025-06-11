import pdb
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, TopKLogitsWarper
from typing import List, Dict

class COTDecoding():

    def __init__(self, load_model, max_new_tokens: int, pattern: str, topk: int, stop: List[str], methods: List[str], template: str = 'standard', device: str = 'cuda'):
        self.model = load_model.model
        self.tokenizer = load_model.tokenizer
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        self.max_new_tokens = max_new_tokens
        self.pattern = pattern
        self.topk = topk
        self.stop = stop
        self.methods = methods
        self.verbose = True
        self.format_question = self.standard_template if template == 'standard' else self.prompt_template
        self.device = device
        if 'cuda' in self.device:
            self.model.to(self.device)


    def extract_methods_bck(self, paths: Dict[str, str]):
        methods = {}
        # greedy decode
        if 'gd' in self.methods:
            methods['gd'] = [path for path in paths if path['k'] == 0]
        # cot-decoding
        if 'cd' in self.methods:
            methods['cd'] = [max(paths[:limit], key=lambda x: x['score']) for limit in range(1, len(paths) + 1)]
        # cot-decoding + self consistency
        if 'cds' in self.methods:
            consistency = {}
            for path in paths:
                if path['answer_span'] not in consistency:
                    consistency[path['answer_span']] = 0
                consistency[path['answer_span']] += path['score']

            major_answer_span = max(consistency, key=consistency.get)
            methods['cds'] = [max([item for item in paths if item['answer_span'] == major_answer_span], key=lambda x: x['score'])]

        return methods

    def extract_methods(self, paths: List[Dict[str, any]]) -> Dict[str, List[Dict[str, any]]]:
        methods = {}
        # Greedy decode (gd)
        if 'gd' in self.methods:
            methods['gd'] = [path for path in paths if path.get('k') == 0]

        # Chain-of-thought decoding (cd)
        if 'cd' in self.methods:
            methods['cd'] = [max(paths[:limit], key=lambda x: x['score']) for limit in range(1, len(paths) + 1)]

        # Chain-of-thought decoding + self-consistency (cds)
        if 'cds' in self.methods:
            consistency = {}
            for path in paths:
                answer_span = path.get('answer_span')
                if answer_span not in consistency:
                    consistency[answer_span] = 0
                consistency[answer_span] += path.get('score', 0)

            if consistency:  # Ensure there's at least one answer_span
                major_answer_span = max(consistency, key=consistency.get)
                methods['cds'] = [
                    max([item for item in paths if item.get('answer_span') == major_answer_span],
                        key=lambda x: x['score'])
                ]

        return methods

    def generate_text(self, prompt: str):
        prompt = self.format_question(prompt)
        topk_tokens = self.get_first_topk_tokens(prompt)
        prompts = [prompt + token for token in topk_tokens['decoded']]
        outputs = self.generate_paths(prompts)
        paths = self.get_paths(topk_tokens, outputs)
        return paths

    def get_paths(self, topk_tokens: Dict[str, any], outputs: List) -> List[Dict[str, any]]:
        paths = []
        for k, output in enumerate(outputs):
            reasoning = topk_tokens['decoded'][k] +  output["outputs"][0]["text"]
            encode = self.tokenizer(reasoning, return_offsets_mapping=True)
            pattern_found = re.findall(self.pattern, reasoning)

            if len(pattern_found):
                last_pattern_span = (reasoning.rfind(pattern_found[-1]), reasoning.rfind(pattern_found[-1]) + len(pattern_found[-1]))
                idx_answer = [i for i, span in enumerate(encode.offset_mapping)
                              if (span[0] >= last_pattern_span[0] and span[1] <= last_pattern_span[1]) or
                                 (span[0] <= last_pattern_span[0] and span[1] >= last_pattern_span[1]) or
                                 (span[0] <= last_pattern_span[0] and span[1] > last_pattern_span[0])]

                token_id = [encode.input_ids[idx] for idx in idx_answer]

                output["outputs"][0]["logprobs"].insert(0, topk_tokens['logprobs'][k])

                filtered_answer = [output for i, output in enumerate(output["outputs"][0]["logprobs"]) if i in idx_answer]

                sum_answer_span_probs = 0
                for logprob_dict in filtered_answer:
                    logprob_list = list(logprob_dict.items())
                    if len(logprob_list) == 2:
                        prob_diff = (torch.exp(torch.tensor([logprob_list[0][1]["logprob"]])) - torch.exp( torch.tensor([logprob_list[1][1]["logprob"]]))).item()
                    else:
                        prob_diff = torch.exp(torch.tensor([logprob_list[0][1]["logprob"]])).item()
                    sum_answer_span_probs += prob_diff
                score = sum_answer_span_probs / len(filtered_answer) if len(filtered_answer) > 0 else 0
                answer_span = self.tokenizer.decode(token_id).strip()
            else:
                score = 0
                answer_span = self.tokenizer.eos_token

            paths.append({'score': score, 'reasoning': reasoning, 'answer_span': answer_span, 'k': k})
        return paths

    @torch.inference_mode()
    def generate_paths(self, prompts: List[str]):
        # Tokenize all prompts at once (batch input)
        inputs = self.tokenizer(prompts, return_tensors='pt', padding=True, truncation=True).to(self.device)

        # Generate multiple output sequences with scores/logits for each token
        output_sequences = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,  # Equivalent to max_tokens=300 in SamplingParams
            num_return_sequences=1,  # Number of output sequences to generate
            do_sample=False,  # Enable sampling to get the same sequences
            top_p=1.0,  # Only sample from top-p cumulative probabilities
            output_scores=True,  # Request scores/logits
            return_dict_in_generate=True,  # Return dictionary with scores
        )
        # Initialize the list to store output results for each prompt
        all_results = []

        # Extract generated sequences, scores, and input information
        sequences = output_sequences.sequences
        scores = output_sequences.scores  # Logits for each generated token
        batch_size = sequences.shape[0]

        # Convert stop tokens from text to token IDs
        stop_token_ids = [self.tokenizer.encode(token, add_special_tokens=False) for token in self.stop]
        pad_token_id = self.tokenizer.pad_token_id
        # For each prompt in the batch
        for i in range(batch_size):
            prompt = prompts[i]
            input_ids = inputs['input_ids'][i]  # Input tokens for this prompt
            # Decode the generated tokens (excluding the prompt part)
            generated_tokens = sequences[i][len(input_ids):]  # Ignore the original prompt tokens
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)


            # Extract top-2 logprobs for the generated tokens
            logprobs = []
            cumulative_logprob = 0.0

            for idx, token_id in enumerate(generated_tokens):
                if token_id == pad_token_id:
                    continue  # Skip pad tokens
                # Compute log probabilities for the current token
                token_logprobs = torch.nn.functional.log_softmax(scores[idx], dim=-1)
                top_logprobs, top_token_ids = torch.topk(token_logprobs[i], k=2)  # Top 2 logprobs

                top_logprobs = top_logprobs.tolist()
                top_token_ids = top_token_ids.tolist()

                # Store the top 2 logprobs and their corresponding tokens
                logprob_dict = {}
                for idx, logprob in enumerate(top_logprobs):
                    logprob_dict.update({top_token_ids[idx]: {"logprob": logprob, "rank": idx, "decoded_token": self.tokenizer.decode([top_token_ids[idx]])}})

                logprobs.append(logprob_dict)
                # Update cumulative logprob with the actual generated token logprob
                cumulative_logprob += token_logprobs[i][token_id].item()

            # Check for stop tokens defined in self.stop
            # truncated_tokens = generated_tokens.tolist()

            # # Handle multi-token stop sequences
            # stop_found = False
            # for stop_token_id in stop_token_ids:
            #     stop_len = len(stop_token_id)
            #     # Slide over the generated tokens and check for subsequences that match stop_token_id
            #     for j in range(len(truncated_tokens) - stop_len + 1):
            #         if truncated_tokens[j:j + stop_len] == stop_token_id:
            #             stop_index = j + stop_len  # Include the full stop sequence in the truncated output
            #             truncated_tokens = truncated_tokens[:stop_index]
            #             stop_found = True
            #             break  # Stop further checks once the stop token sequence is found
            #     if stop_found:
            #         break  # Stop checking other stop sequences if one has been found
            #
            # Structure the result for this prompt
            result = {
                "request_id": i + 1,
                "prompt": prompt,
                "prompt_token_ids": input_ids.tolist(),
                "outputs": [{
                    "text": generated_text,
                    "token_ids": self.tokenizer.encode(generated_text, add_special_tokens=False),
                    "logprobs": logprobs,
                    "cumulative_logprob": cumulative_logprob,
                    # "finish_reason": "stop" if stop_found else "max_tokens"
                }]
            }
            all_results.append(result)
        return all_results

    @torch.inference_mode()
    def get_first_topk_tokens(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs,)
        logits = outputs.logits[:, -1, :]  # Get the logits for the next token
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        topk_logprobs, topk_indices = torch.topk(log_probs, k=self.topk, dim=-1)
        decoded_tokens = [self.tokenizer.decode([idx.item()]) for idx in topk_indices[0]]
        logprobs_list = []

        for idx in range(len(topk_indices[0])):
            token_id = topk_indices[0][idx].item()
            logprob = topk_logprobs[0][idx].item()
            decoded_token = decoded_tokens[idx]
            logprobs_list.append({token_id: {'logprob': logprob, 'rank': idx, 'decoded_token': decoded_token}})

        topk_tokens = {
            'decoded': decoded_tokens,
            'probs': [logprob for logprob in torch.exp(topk_logprobs[0]).tolist()],
            'token_id': [idx.item() for idx in topk_indices[0]],
            'logprobs': logprobs_list
        }
        return topk_tokens

    def standard_template(self, prompt: str):
        return f"Q: {prompt}\nA:"

    def prompt_template(self, prompt: str):
        return f"Q: {prompt}\nA: Let's think step by step."