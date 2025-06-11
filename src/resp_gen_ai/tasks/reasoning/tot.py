import os
import re
import json
import sympy
import pandas as pd
import itertools
import pdb
import numpy as np
from functools import partial
from resp_gen_ai.datasets.reasoning.reasoning_datasets import ReasoningLMDataset

import backoff
import openai


# TODO: Replace these with actual imports from the GPT module
# from tot.models import gpt
# -------------------------------------------------------------------------------------
# Base Task Class
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------
# get_task Function
# -------------------------------------------------------------------------------------
def get_task(name, dataset: ReasoningLMDataset):
    """
    Create the appropriate Task subclass, pulling the correct data
    from `dataset` (a ReasoningLMDataset instance).
    """
    if name == 'game24':
        # Suppose we stored puzzle strings in dataset.data['puzzle_column']
        file_data = dataset.data['puzzle_column'].tolist()
        return Game24Task(file_data)

    elif name == 'text':
        # Suppose text data is in dataset.data (already a list of strings)
        file_data = dataset.data
        return TextTask(file_data)

    elif name == 'crosswords':
        # Suppose crosswords are in dataset.data,
        # shaped like a list of (clues, board) for each puzzle
        file_data = dataset.data  # e.g. a list of puzzles
        return MiniCrosswordsTask(file_data)

    else:
        raise NotImplementedError(f"The task '{name}' is not implemented.")

class Task:
    """
    Base Task class providing the general interface:
      - __len__: how many samples the task has
      - get_input: returns the input (prompt) for a specific index
      - test_output: evaluates how good or bad the output is (optional reward)
    """
    def __init__(self):
        pass

    def __len__(self) -> int:
        """
        Return the number of items in this Task.
        """
        raise NotImplementedError

    def get_input(self, idx: int) -> str:
        """
        Return the input text/string for the given index.
        """
        raise NotImplementedError

    def test_output(self, idx: int, output: str):
        """
        Evaluate the output for a given index's input.
        """
        raise NotImplementedError
# -------------------------------------------------------------------------------------
# gpt and Prompt Imports (assuming they are local modules)
# -------------------------------------------------------------------------------------


# Crosswords prompts
try:
    from resp_gen_ai.prompts.tot_prompts.crosswords import value_prompt, standard_prompt, cot_prompt, propose_prompt
except ImportError:
    # Provide some placeholders if the import is not available
    value_prompt = "Value prompt placeholder: {input}"
    standard_prompt = "Standard prompt placeholder: {input}"
    cot_prompt = "Chain of Thought placeholder: {input}"
    propose_prompt = "Propose prompt placeholder: {input}"

# Text prompts
try:
    from resp_gen_ai.prompts.tot_prompts.text import score_prompt, vote_prompt, compare_prompt, standard_prompt, cot_prompt
except ImportError:
    # Provide some placeholders if the import is not available
    score_prompt = "Score prompt placeholder"
    vote_prompt = "Vote prompt placeholder"
    compare_prompt = "Compare prompt placeholder"

# Game24 prompts
try:
    from resp_gen_ai.prompts.tot_prompts.game24 import value_prompt as game24_value_prompt
    from resp_gen_ai.prompts.tot_prompts.game24 import value_last_step_prompt, standard_prompt as game24_standard_prompt
    from resp_gen_ai.prompts.tot_prompts.game24 import cot_prompt as game24_cot_prompt, propose_prompt as game24_propose_prompt
except ImportError:
    # Provide some placeholders if the import is not available
    game24_value_prompt = "Game24 Value prompt placeholder"
    value_last_step_prompt = "Last Step Value Prompt placeholder: {answer}"
    game24_standard_prompt = "Game24 Standard prompt placeholder"
    game24_cot_prompt = "Game24 CoT prompt placeholder"
    game24_propose_prompt = "Game24 Propose prompt placeholder"

# -------------------------------------------------------------------------------------
# MiniCrosswords Environment
# -------------------------------------------------------------------------------------
class MiniCrosswordsEnv:
    """
    Environment that loads a mini-crossword puzzle from a JSON file and
    keeps track of the puzzle's board state, answers, and steps taken.
    """
    def __init__(self, file_data):
        self.file_data = file_data
        self.n = len(self.file_data)
        self.cache = {}
        self.idx = None
        self.times = 0
        # Caches the results of GPT value calls so we don't re-query
        self.prompt_status_cache = {}

    def __len__(self):
        """
        Number of crossword samples in the loaded file.
        """
        return self.n

    def reset(self, idx, board=None, status=None, steps=None):
        """
        Reset the environment state for puzzle at index = idx.
          - board: custom board for partial solutions
          - status: current fill status of each row/column
          - steps: how many steps have been taken so far
        """
        self.idx = idx
        # Each entry in self.data is a tuple: (list_of_clues, correct_board)
        self.data, self.board_gt = self.file_data[idx]

        # Initialize a blank board of 25 cells (5x5), underscores denote empty.
        self.board = ['_'] * 25
        self.ans = ['_____'] * 10  # Each row and column answer
        self.ans_gt = self.get_ans(self.board_gt)

        # 0: unfilled; 1: filled; 2: changed after being filled
        self.status = [0] * 10
        self.steps = 0

        # If custom states (board, status, steps) are provided, use them.
        if board is not None:
            self.board = board
            self.ans = self.get_ans(self.board)
        if status is not None:
            self.status = status
        if steps is not None:
            self.steps = steps

        return self.render()

    def prompt_status(self):
        """
        Queries GPT to evaluate how many words it thinks are sure, maybe, or impossible
        among all partially filled answers. Only partially filled answers with fewer
        than 4 blanks are queried to reduce noise.
        """
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}

        for ans, data, st in zip(self.ans, self.data, self.status):
            # Skip if mostly blank or if not unfilled
            if ans.count('_') >= 4:
                continue
            # Build prompt line
            ans_str = ' '.join(ans.lower())
            line = f'{data}: {ans_str}'
            prompt = value_prompt.format(input=line)

            # Use cached result if we have it
            if prompt in self.prompt_status_cache:
                res = self.prompt_status_cache[prompt]
            else:
                res = gpt(prompt)[0]
                self.prompt_status_cache[prompt] = res

            # Parse the last line of GPT response as the label
            res = res.split('\n')[-1].strip()
            if res in count:
                count[res] += 1

        return count

    def render_gt_board(self):
        """
        Returns a string showing the ground truth (correct) board.
        """
        s = "GT Board:\n"
        for i in range(5):
            s += ' '.join(self.board_gt[i * 5:(i + 1) * 5]) + '\n'
        return s

    def render_board(self):
        """
        Returns a string representing the current board.
        """
        s = "Current Board:\n"
        for i in range(5):
            s += ''.join(self.board[i * 5:(i + 1) * 5]) + '\n'
        return s

    def render_clues(self, status=None):
        """
        Render the crossword clues.
        If status is not None, only clues with that fill status are shown.
        Horizontal are 0-4, vertical are 5-9.
        """
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += f'h{i + 1}. {self.data[i]}\n'
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += f'v{i - 5 + 1}. {self.data[i]}\n'
        return s

    def render_ans(self, status=None):
        """
        Render the answers, grouped by horizontal and vertical.
        If status is not None, only show those with that fill status.
        """
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += f'h{i + 1}. {self.data[i]}: {self.ans[i]}\n'
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += f'v{i - 5 + 1}. {self.data[i]}: {self.ans[i]}\n'
        return s

    def render_gt_ans(self, status=None):
        """
        Render the ground-truth answers for each clue, grouped by horizontal and vertical.
        If status is not None, only show those with that fill status.
        """
        s = ""
        for i in range(5):
            if status is None or self.status[i] == status:
                s += f'h{i + 1}. {self.data[i]}: {self.ans_gt[i]}\n'
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += f'v{i - 5 + 1}. {self.data[i]}: {self.ans_gt[i]}\n'
        return s

    def render(self, status=True):
        """
        Render the current board and, if status=True, also render
        which clues are unfilled, filled, or changed.
        """
        if status:
            return (
                self.render_board()
                + '\nUnfilled:\n'
                + self.render_ans(status=0)
                + '\nFilled:\n'
                + self.render_ans(status=1)
                + '\nChanged:\n'
                + self.render_ans(status=2)
            )
        else:
            return self.render_board() + '\n' + self.render_ans()

    def get_ans(self, board):
        """
        Extract the 5 horizontal and 5 vertical answers from a 5x5 board.
        """
        ans = [''] * 10
        # Horizontal answers
        for i in range(5):
            ans[i] = ''.join(board[i * 5:(i + 1) * 5])
        # Vertical answers
        for i in range(5):
            ans[i + 5] = ''.join(board[i::5])
        return ans

    def step(self, action):
        """
        Accepts an action of the form 'h1. apple' or 'v3. crane'
        and updates the puzzle's board accordingly.
        Returns:
         - rendered view
         - if the puzzle is fully solved (boolean)
         - if done (either solved or reached 20 steps)
         - info dict with letter-accuracy, word-accuracy, and game completion
        """
        self.steps += 1

        # Attempt to parse the action
        action = action.split('\n')[-1]
        action = action.split('. ')
        if len(action) != 2:
            return 'Invalid! Format should be like "h1. apple"', 0, False, {}

        pos, word = action
        if len(word) != 5:
            return 'Invalid! Word should have 5 letters.', 0, False, {}

        # Horizontal or vertical placement
        if pos.startswith('h'):
            idx = int(pos[1:]) - 1
            self.board[idx * 5:(idx + 1) * 5] = list(word.upper())
        elif pos.startswith('v'):
            idx = int(pos[1:]) - 1
            self.board[idx::5] = list(word.upper())
            idx += 5  # Adjust for vertical clue index
        else:
            return 'Invalid! Position should be h1-h5 or v1-v5', 0, False, {}

        # Check if any previously filled letter changed
        self.new_ans = self.get_ans(self.board)
        self.status = [
            2 if any(letter != new_letter and letter != '_'
                     for letter, new_letter in zip(ans, new_ans)) else st
            for st, ans, new_ans in zip(self.status, self.ans, self.new_ans)
        ]

        # Mark the newly filled line as status = 1
        self.status[idx] = 1
        self.ans = self.new_ans

        # Check completion:
        r_all = (self.board == self.board_gt)  # Fully correct board?
        r_letter = sum(a == b for a, b in zip(self.board, self.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(self.ans, self.ans_gt)) / 10

        # Done if fully correct or reached 20 steps
        return (
            self.render(),
            r_all,
            (r_all or self.steps >= 20),
            {'r_letter': r_letter, 'r_word': r_word, 'r_game': r_all}
        )

# -------------------------------------------------------------------------------------
# MiniCrosswords Task
# -------------------------------------------------------------------------------------
class MiniCrosswordsTask(Task):
    """
    Task wrapper for the MiniCrosswordsEnv.
    Provides standard Task interface and additional prompt/answer logic.
    """
    def __init__(self, file_data):
        super().__init__()
        self.env = MiniCrosswordsEnv(file_data)  # underlying environment
        self.xs = []
        # Pre-load all puzzles' clues
        for idx in range(len(self.env)):
            self.env.reset(idx)
            self.xs.append(self.env.render_clues())
        # By default, we allow 10 steps to solve
        self.steps = 10
        # Cache proposals
        self.cache_proposals = {}

    def __len__(self) -> int:
        return len(self.env)

    def get_input(self, idx: int) -> str:
        """
        Return the textual clues for puzzle at index = idx.
        """
        self.env.reset(idx)
        return self.env.render_clues()

    def test_output(self, idx: int, output: str):
        """
        Evaluate output by simulating the environment steps using the
        last 5 lines as words for each horizontal row.
        """
        self.env.reset(idx)
        # Here we assume the final 5 lines fill horizontal lines h1-h5
        output = output.split('Output:\n')[-1]
        info = {'r_word': 0, 'r_letter': 0, 'r_game': 0}
        for i, line in enumerate(output.strip().split('\n')[-5:], 1):
            letters = line.split(' ')[:5]
            word = ''.join(letters)
            word = word + '_' * (5 - len(word))  # ensure 5 letters
            action = f'h{i}. {word}'
            _, _, _, info = self.env.step(action)

        # We store 'r' as r_word in the info, consistent with other tasks
        info['r'] = info['r_word']
        return info

    def set_status(self, x: str, y: str):
        """
        Helper that resets environment to the puzzle matching x,
        then steps with output y.
        """
        idx = self.xs.index(x)
        self.test_output(idx, y)  # update self.env

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        return cot_prompt.format(input=x) + y

    def propose_prompt_wrap(self, x: str, y: str = '') -> str:
        """
        If x and y are given, set puzzle status first, then generate a prompt
        to propose the next step.
        """
        self.set_status(x, y)
        return propose_prompt.format(input=self.env.render())

    def propose_outputs_unwrap(self, x: str, y: str, outputs: list, n_max_propose: int) -> list:
        """
        Parse model outputs to gather lines in the format:
          h1. apple (confidence)
        and keep top n proposals by confidence sum.
        """
        confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}
        proposals_to_scores = {}

        # Regular expression to parse lines like "h1. APPLE (high)"
        pattern = r'^([hv][1-5])\. ([a-zA-Z]{5}) \((certain|high|medium|low)\).*$'

        for output in outputs:
            lines = output.split('\n')
            for line in lines:
                match = re.match(pattern, line)
                if match:
                    clue = match.group(1).lower() + '. ' + match.group(2).lower()
                    conf_str = match.group(3)
                    score = confidence_to_value.get(conf_str, 0)
                    proposals_to_scores[clue] = proposals_to_scores.get(clue, 0) + score

        # Sort proposals by descending total confidence
        proposals = sorted(proposals_to_scores.items(), key=lambda x: x[1], reverse=True)

        # Keep only top n if n_max_propose != -1
        if n_max_propose != -1:
            proposals = proposals[:n_max_propose]

        # Rebuild them in y-append style (one per line)
        proposals = [y + p[0] + '\n' for p in proposals]
        self.cache_proposals[(x, y, n_max_propose)] = proposals
        return proposals

    def evaluate(self, x: str, y: str, n_evaluate_sample: int) -> int:
        """
        Evaluate the fill status of partially completed puzzle.
        n_evaluate_sample: how many times we want to re-sample or evaluate (ad-hoc).
        """
        self.set_status(x, y)
        assert n_evaluate_sample == 1, "Only one sample currently supported."

        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        for ans, data, status in zip(self.env.ans, self.env.data, self.env.status):
            # Skip if mostly blank
            if ans.count('_') >= 4:
                continue
            ans_str = ' '.join(ans.lower())
            line = f'{data}: {ans_str}'
            prompt = value_prompt.format(input=line)
            res = gpt(prompt)[0]
            print(line, res, '')
            res = res.split('\n')[-1].strip()
            if res in count:
                count[res] += 1

        print(count)
        return count

# -------------------------------------------------------------------------------------
# Text Task
# -------------------------------------------------------------------------------------
class TextTask(Task):
    """
    A generic text-generation / text-completion task.
    Each line in the given file is a separate "prompt". The output is
    a generated passage, and the reward can be based on coherence (for example).
    """
    def __init__(self,file_data):
        super().__init__()
        self.file_data = file_data
        self.steps = 2
        # Potential stops or tokens that end the generation
        self.stops = ['\nPassage:\n', None]

    def __len__(self) -> int:
        return len(self.file_data)

    def get_input(self, idx: int) -> str:
        """
        Return the text prompt for the line at index = idx.
        """
        return self.file_data[idx]

    def test_output(self, idx: int, output: str):
        """
        Example test_output that attempts to parse "coherency score"
        from multiple GPT responses.
        """
        output_segment = output.split('Passage:\n')[-1]
        prompt = f"{score_prompt}{output_segment}"
        score_outputs = gpt(prompt, num_return_sequences=5, model='gpt-4')

        scores = []
        pattern = r".*coherency score is (\d+).*"
        for score_output in score_outputs:
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'No match for score in: {score_output}')

        info = {
            'rs': scores,
            'r': sum(scores) / len(scores) if scores else 0
        }
        return info

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        """
        Build a prompt that compares multiple candidate completions
        (Choice 1, Choice 2, etc.).
        """
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            prompt += f'Choice {i}:\n{y}\n'
        return prompt

    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        """
        Tally up the votes for each candidate.
        """
        vote_results = [0] * n_candidates
        pattern = r".*best choice is .*(\d+).*"
        for vote_output in vote_outputs:
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if 0 <= vote < n_candidates:
                    vote_results[vote] += 1
            else:
                print(f'Vote no match: {vote_output}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        """
        Compare exactly two candidate passages for coherence.
        """
        assert len(ys) == 2, 'compare_prompt_wrap supports only 2 candidates'
        passage_1 = ys[0].split('Passage:\n')[-1]
        passage_2 = ys[1].split('Passage:\n')[-1]
        return compare_prompt + f'Passage 1:\n{passage_1}\n\nPassage 2:\n{passage_2}\n'

    @staticmethod
    def compare_output_unwrap(compare_output: str):
        """
        Parse which passage is declared more coherent (1 or 2),
        or if they are similarly coherent.
        """
        if 'more coherent passage is 1' in compare_output:
            return 0
        elif 'more coherent passage is 2' in compare_output:
            return 1
        elif 'two passages are similarly coherent' in compare_output:
            return 0.5
        else:
            print(f'Compare no match: {compare_output}')
            return -1

# -------------------------------------------------------------------------------------
# Game24 Task
# -------------------------------------------------------------------------------------
def get_current_numbers(y: str) -> str:
    """
    Extracts the 'left: X' portion from the last line to see
    which numbers remain to be used in the puzzle.
    """
    last_line = y.strip().split('\n')[-1]
    return last_line.split('left: ')[-1].split(')')[0]

class Game24Task(Task):
    """
    A puzzle-like task for checking if four numbers can be used
    (with arithmetic operations) to reach the number 24.
    """
    def __init__(self, file_data):
        super().__init__()
        self.file_data =  file_data
        self.value_cache = {}
        # Allow up to 4 steps or lines in typical 24-solution expansions
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.file_data)

    def get_input(self, idx: int) -> str:
        """
        Return the 4 numbers (the puzzle) for a given index.
        """
        return self.file_data[idx]

    def test_output(self, idx: int, output: str):
        """
        Test if the final expression actually simplifies to 24.
        We expect the last line to contain 'Answer: <expression> = 24'.
        """
        # Grab the final expression
        expression_line = output.strip().split('\n')[-1].lower()
        expression_line = expression_line.replace('answer: ', '')

        # Remove the "= ..." part to isolate expression
        expression = expression_line.split('=')[0].strip()
        # Check that the numeric components match the puzzle's digits
        puzzle_numbers = sorted(re.findall(r'\d+', self.file_data[idx]))
        used_numbers = sorted(re.findall(r'\d+', expression))

        if puzzle_numbers != used_numbers:
            return {'r': 0}  # Mismatch in digits used

        # Try to evaluate the expression
        try:
            if sympy.simplify(expression) == 24:
                return {'r': 1}
            else:
                return {'r': 0}
        except Exception:
            return {'r': 0}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        return game24_standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        return game24_cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = '') -> str:
        """
        If the puzzle is not yet at 24, propose next step using the leftover numbers.
        """
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            # If we already have 24, just finalize
            prompt = game24_cot_prompt.format(input=x) + 'Steps:' + y
        else:
            prompt = game24_propose_prompt.format(input=current_numbers)
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        """
        Creates a "value" prompt that checks if the partial or final expression
        is correct or feasible.
        """
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:
            # Means it's presumably the last step with an answer
            ans = last_line.lower().replace('answer: ', '')
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return game24_value_prompt.format(input=current_numbers)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        """
        Maps model-supplied labels (like "impossible", "likely", "sure")
        to numeric scores. Summation of scores across multiple outputs.
        """
        steps = y.strip().split('\n')
        # If we have 4 lines but no final 'Answer: ', treat as partial
        if len(steps) == 4 and 'answer' not in y.lower():
            return 0

        # Possible confidence labels and their numeric scores
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        # We get the last line of each GPT output and compare
        value_names = [resp.split('\n')[-1].strip() for resp in value_outputs]

        # Add up the assigned values
        total_value = 0
        for name in value_names:
            total_value += value_map.get(name, 0)

        return total_value

def get_value(task, x, y, n_evaluate_sample, cache_value=True):
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, num_return_sequences=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value


def get_values(task, x, ys, n_evaluate_sample, cache_value=True):
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values


def get_votes(task, x, ys, n_evaluate_sample):
    vote_prompt = task.vote_prompt_wrap(x, ys)
    vote_outputs = gpt(vote_prompt, num_return_sequences=n_evaluate_sample, stop=None)
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values


def get_proposals(task, x, y):
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, num_return_sequences=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]


def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):
    if prompt_sample == 'standard':
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, num_return_sequences=n_generate_sample, stop=stop)
    return [y + _ for _ in samples]


def solve(args, model, task, idx, to_print=True):
    global gpt
    gpt = model.generate
    x = task.get_input(idx)  # input
    ys = ['']  # current output candidates
    infos = []
    for step in range(task.steps):
        # generation
        if args.method_generate == 'sample':
            new_ys = [
                get_samples(task, x, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step])
                for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))
        # evaluation
        if args.method_evaluate == 'vote':
            values = get_votes(task, x, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x, new_ys, args.n_evaluate_sample)

        # selection
        if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]
        select_new_ys = [new_ys[select_id] for select_id in select_ids]

        # log
        if to_print:
            sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}\n-- choices --: {select_new_ys}\n')

        infos.append(
            {'step': step, 'x': x, 'ys': ys, 'new_ys': new_ys, 'values': values, 'select_new_ys': select_new_ys})
        ys = select_new_ys

    if to_print:
        print(ys)
    return ys, {'steps': infos}

def naive_solve(args, task, idx, to_print=True):
    global gpt
    print(gpt)
    x = task.get_input(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
