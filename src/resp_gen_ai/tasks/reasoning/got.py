from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Union
from enum import Enum
import json
import logging
import pdb
import itertools
from typing import List
from resp_gen_ai.models.llms.base import LLMBaseModel

class Prompter(ABC):
    """
    Abstract base class that defines the interface for all prompters.
    Prompters are used to generate the prompts for the language models.
    """

    @abstractmethod
    def aggregation_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a aggregation prompt for the language model.

        :param state_dicts: The thought states that should be aggregated.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The aggregation prompt.
        :rtype: str
        """
        pass

    @abstractmethod
    def improve_prompt(self, **kwargs) -> str:
        """
        Generate an improve prompt for the language model.
        The thought state is unpacked to allow for additional keyword arguments
        and concrete implementations to specify required arguments explicitly.

        :param kwargs: Additional keyword arguments.
        :return: The improve prompt.
        :rtype: str
        """
        pass

    @abstractmethod
    def generate_prompt(self, num_branches: int, **kwargs) -> str:
        """
        Generate a generate prompt for the language model.
        The thought state is unpacked to allow for additional keyword arguments
        and concrete implementations to specify required arguments explicitly.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        """
        pass

    @abstractmethod
    def validation_prompt(self, **kwargs) -> str:
        """
        Generate a validation prompt for the language model.
        The thought state is unpacked to allow for additional keyword arguments
        and concrete implementations to specify required arguments explicitly.

        :param kwargs: Additional keyword arguments.
        :return: The validation prompt.
        :rtype: str
        """
        pass

    @abstractmethod
    def score_prompt(self, state_dicts: List[Dict], **kwargs) -> str:
        """
        Generate a score prompt for the language model.

        :param state_dicts: The thought states that should be scored,
                            if more than one, they should be scored together.
        :type state_dicts: List[Dict]
        :param kwargs: Additional keyword arguments.
        :return: The score prompt.
        :rtype: str
        """
        pass

class Parser(ABC):
    """
    Abstract base class that defines the interface for all parsers.
    Parsers are used to parse the responses from the language models.
    """

    @abstractmethod
    def parse_aggregation_answer(
        self, states: List[Dict], texts: List[str]
    ) -> Union[Dict, List[Dict]]:
        """
        Parse the response from the language model for a aggregation prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the response from the language model.
        :rtype: Union[Dict, List[Dict]]
        """
        pass

    @abstractmethod
    def parse_improve_answer(self, state: Dict, texts: List[str]) -> Dict:
        """
        Parse the response from the language model for an improve prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought state after parsing the response from the language model.
        :rtype: Dict
        """
        pass

    @abstractmethod
    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the response from the language model.
        :rtype: List[Dict]
        """
        pass

    @abstractmethod
    def parse_validation_answer(self, state: Dict, texts: List[str]) -> bool:
        """
        Parse the response from the language model for a validation prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: Whether the thought state is valid or not.
        :rtype: bool
        """
        pass

    @abstractmethod
    def parse_score_answer(self, states: List[Dict], texts: List[str]) -> List[float]:
        """
        Parse the response from the language model for a score prompt.

        :param states: The thought states used to generate the prompt.
        :type states: List[Dict]
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The scores for the thought states.
        :rtype: List[float]
        """
        pass

class GraphOfOperations:
    """
    Represents the Graph of Operations, which prescribes the execution plan of thought operations.
    """

    def __init__(self) -> None:
        """
        Initializes a new Graph of Operations instance with empty operations, roots, and leaves.
        The roots are the entry points in the graph with no predecessors.
        The leaves are the exit points in the graph with no successors.
        """
        self.operations: List[Operation] = []
        self.roots: List[Operation] = []
        self.leaves: List[Operation] = []

    def append_operation(self, operation: Operation) -> None:
        """
        Appends an operation to all leaves in the graph and updates the relationships.

        :param operation: The operation to append.
        :type operation: Operation
        """
        self.operations.append(operation)

        if len(self.roots) == 0:
            self.roots = [operation]
        else:
            for leave in self.leaves:
                leave.add_successor(operation)

        self.leaves = [operation]

    def add_operation(self, operation: Operation) -> None:
        """
        Add an operation to the graph considering its predecessors and successors.
        Adjust roots and leaves based on the added operation's position within the graph.

        :param operation: The operation to add.
        :type operation: Operation
        """
        self.operations.append(operation)
        if len(self.roots) == 0:
            self.roots = [operation]
            self.leaves = [operation]
            assert (
                len(operation.predecessors) == 0
            ), "First operation should have no predecessors"
        else:
            if len(operation.predecessors) == 0:
                self.roots.append(operation)
            for predecessor in operation.predecessors:
                if predecessor in self.leaves:
                    self.leaves.remove(predecessor)
            if len(operation.successors) == 0:
                self.leaves.append(operation)

class Controller:
    """
    Controller class to manage the execution flow of the Graph of Operations,
    generating the Graph Reasoning State.
    This involves language models, graph operations, prompting, and parsing.
    """

    def __init__(
        self,
        lm: AbstractLanguageModel,
        graph: GraphOfOperations,
        prompter: Prompter,
        parser: Parser,
        problem_parameters: dict,
    ) -> None:
        """
        Initialize the Controller instance with the language model,
        operations graph, prompter, parser, and problem parameters.

        :param lm: An instance of the AbstractLanguageModel.
        :type lm: AbstractLanguageModel
        :param graph: The Graph of Operations to be executed.
        :type graph: OperationsGraph
        :param prompter: An instance of the Prompter class, used to generate prompts.
        :type prompter: Prompter
        :param parser: An instance of the Parser class, used to parse responses.
        :type parser: Parser
        :param problem_parameters: Initial parameters/state of the problem.
        :type problem_parameters: dict
        """
        self.logger = logging.getLogger(self.__class__.__module__)
        self.lm = lm
        self.graph = graph
        self.prompter = prompter
        self.parser = parser
        self.problem_parameters = problem_parameters
        self.run_executed = False

    def run(self) -> None:
        """
        Run the controller and execute the operations from the Graph of
        Operations based on their readiness.
        Ensures the program is in a valid state before execution.
        :raises AssertionError: If the Graph of Operation has no roots.
        :raises AssertionError: If the successor of an operation is not in the Graph of Operations.
        """
        self.logger.debug("Checking that the program is in a valid state")
        assert self.graph.roots is not None, "The operations graph has no root"
        self.logger.debug("The program is in a valid state")
        
        execution_queue = [
            operation
            for operation in self.graph.operations
            if operation.can_be_executed()
        ]

        while len(execution_queue) > 0:
            current_operation = execution_queue.pop(0)
            self.logger.info("Executing operation %s", current_operation.operation_type)
            current_operation.execute(
                self.lm, self.prompter, self.parser, **self.problem_parameters
            )
            self.logger.info("Operation %s executed", current_operation.operation_type)
            for operation in current_operation.successors:
                assert (
                    operation in self.graph.operations
                ), "The successor of an operation is not in the operations graph"
                if operation.can_be_executed():
                    execution_queue.append(operation)
        self.logger.info("All operations executed")
        self.run_executed = True

    def get_final_thoughts(self) -> List[List[Thought]]:
        """
        Retrieve the final thoughts after all operations have been executed.

        :return: List of thoughts for each operation in the graph's leaves.
        :rtype: List[List[Thought]]
        :raises AssertionError: If the `run` method hasn't been executed yet.
        """
        assert self.run_executed, "The run method has not been executed"
        return [operation.get_thoughts() for operation in self.graph.leaves]

    def output_graph(self, path: str) -> None:
        """
        Serialize the state and results of the operations graph to a JSON file.

        :param path: The path to the output file.
        :type path: str
        """
        output = []
        for operation in self.graph.operations:
            operation_serialized = {
                "operation": operation.operation_type.name,
                "thoughts": [thought.state for thought in operation.get_thoughts()],
            }
            if any([thought.scored for thought in operation.get_thoughts()]):
                operation_serialized["scored"] = [
                    thought.scored for thought in operation.get_thoughts()
                ]
                operation_serialized["scores"] = [
                    thought.score for thought in operation.get_thoughts()
                ]
            if any([thought.validated for thought in operation.get_thoughts()]):
                operation_serialized["validated"] = [
                    thought.validated for thought in operation.get_thoughts()
                ]
                operation_serialized["validity"] = [
                    thought.valid for thought in operation.get_thoughts()
                ]
            if any(
                [
                    thought.compared_to_ground_truth
                    for thought in operation.get_thoughts()
                ]
            ):
                operation_serialized["compared_to_ground_truth"] = [
                    thought.compared_to_ground_truth
                    for thought in operation.get_thoughts()
                ]
                operation_serialized["problem_solved"] = [
                    thought.solved for thought in operation.get_thoughts()
                ]
            output.append(operation_serialized)

        # output.append(
        #     {
        #         "prompt_tokens": self.lm.prompt_tokens,
        #         "completion_tokens": self.lm.completion_tokens,
        #         "cost": self.lm.cost,
        #     }
        # )

        with open(path, "w") as file:
            file.write(json.dumps(output, indent=2))

class OperationType(Enum):
    """
    Enum to represent different operation types that can be used as unique identifiers.
    """

    score: int = 0
    validate_and_improve: int = 1
    generate: int = 2
    improve: int = 3
    aggregate: int = 4
    keep_best_n: int = 5
    keep_valid: int = 6
    ground_truth_evaluator: int = 7
    selector: int = 8


class Operation(ABC):
    """
    Abstract base class that defines the interface for all operations.
    """

    _ids: Iterator[int] = itertools.count(0)

    operation_type: OperationType = None

    def __init__(self) -> None:
        """
        Initializes a new Operation instance with a unique id, and empty predecessors and successors.
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Operation._ids)
        self.predecessors: List[Operation] = []
        self.successors: List[Operation] = []
        self.executed: bool = False

    def can_be_executed(self) -> bool:
        """
        Checks if the operation can be executed based on its predecessors.

        :return: True if all predecessors have been executed, False otherwise.
        :rtype: bool
        """
        return all(predecessor.executed for predecessor in self.predecessors)

    def get_previous_thoughts(self) -> List[Thought]:
        """
        Iterates over all predecessors and aggregates their thoughts.

        :return: A list of all thoughts from the predecessors.
        :rtype: List[Thought]
        """
        previous_thoughts: List[Thought] = [
            thought
            for predecessor in self.predecessors
            for thought in predecessor.get_thoughts()
        ]

        return previous_thoughts

    def add_predecessor(self, operation: Operation) -> None:
        """
        Add a preceding operation and update the relationships.

        :param operation: The operation to be set as a predecessor.
        :type operation: Operation
        """
        self.predecessors.append(operation)
        operation.successors.append(self)

    def add_successor(self, operation: Operation) -> None:
        """
        Add a succeeding operation and update the relationships.

        :param operation: The operation to be set as a successor.
        :type operation: Operation
        """
        self.successors.append(operation)
        operation.predecessors.append(self)

    def execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Execute the operation, assuring that all predecessors have been executed.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If not all predecessors have been executed.
        """
        assert self.can_be_executed(), "Not all predecessors have been executed"
        self.logger.info(
            "Executing operation %d of type %s", self.id, self.operation_type
        )
        self._execute(lm, prompter, parser, **kwargs)
        self.logger.debug("Operation %d executed", self.id)
        self.executed = True

    @abstractmethod
    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Abstract method for the actual execution of the operation.
        This should be implemented in derived classes.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        pass

    @abstractmethod
    def get_thoughts(self) -> List[Thought]:
        """
        Abstract method to retrieve the thoughts associated with the operation.
        This should be implemented in derived classes.

        :return: List of associated thoughts.
        :rtype: List[Thought]
        """
        pass


class Score(Operation):
    """
    Operation to score thoughts.
    """

    operation_type: OperationType = OperationType.score

    def __init__(
        self,
        num_samples: int = 1,
        combined_scoring: bool = False,
        scoring_function: Callable[
            [Union[List[Dict], Dict]], Union[List[float], float]
        ] = None,
    ) -> None:
        """
        Initializes a new Score operation.

        :param num_samples: Number of samples to use for scoring. Defaults to 1.
        :type num_samples: int
        :param combined_scoring: Whether to score all thoughts together or individually. Defaults to False.
        :type combined_scoring: bool
        :param scoring_function: A function to score thoughts (if not using LM). Defaults to None.
        :type scoring_function: Takes a list of thought states or a single thought state and
                                returns a list of scores or a single score.
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.combined_scoring: bool = combined_scoring
        self.thoughts: List[Thought] = []
        self.scoring_function: Callable[
            [Union[List[Dict], Dict]], Union[List[float], float]
        ] = scoring_function

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation.

        :return: List of scored thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the scoring operation by scoring the thoughts from the predecessors.
        If combined scoring is used, the thoughts are scored together, otherwise individually.
        If a scoring function is provided, it is used, otherwise the LM is prompted.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert (
            len(self.predecessors) > 0
        ), "Score operation needs at least one predecessor"

        if self.combined_scoring:
            previous_thoughts_states = [thought.state for thought in previous_thoughts]
            if self.scoring_function is not None:
                self.logger.debug(
                    "Using scoring function %s to score states", self.scoring_function
                )
                scores = self.scoring_function(previous_thoughts_states)
            else:
                prompt = prompter.score_prompt(previous_thoughts_states)
                self.logger.debug("Prompt for LM: %s", prompt)

                responses = lm.generate(prompt, num_return_sequences=self.num_samples)
                self.logger.debug("Responses from LM: %s", responses)
                scores = parser.parse_score_answer(previous_thoughts_states, responses)
            for thought, score in zip(previous_thoughts, scores):
                new_thought = Thought.from_thought(thought)
                new_thought.score = score
                self.thoughts.append(new_thought)
        else:
            for thought in previous_thoughts:
                new_thought = Thought.from_thought(thought)
                if self.scoring_function is not None:
                    self.logger.debug(
                        "Using scoring function %s to score state",
                        self.scoring_function,
                    )
                    score = self.scoring_function(thought.state)
                else:
                    prompt = prompter.score_prompt([thought.state])
                    self.logger.debug("Prompt for LM: %s", prompt)

                    responses = lm.generate(prompt, num_return_sequences=self.num_samples)
                    self.logger.debug("Responses from LM: %s", responses)
                    score = parser.parse_score_answer([thought.state], responses)[0]

                new_thought.score = score
                self.thoughts.append(new_thought)

        self.logger.info(
            "Score operation %d scored %d thoughts",
            self.id,
            len(self.thoughts),
        )


class ValidateAndImprove(Operation):
    """
    Operation to validate and improve thoughts.
    """

    operation_type: OperationType = OperationType.validate_and_improve

    def __init__(
        self,
        num_samples: int = 1,
        improve: bool = True,
        num_tries: int = 3,
        validate_function: Callable[[Dict], bool] = None,
    ) -> None:
        """
        Initializes a new ValidateAndImprove operation.

        :param num_samples: Number of samples to use for validation. Defaults to 1.
        :type num_samples: int
        :param improve: Whether to improve the thought if it is not valid. Defaults to True.
        :type improve: bool
        :param num_tries: Number of tries to improve the thought before giving up. Defaults to 3.
        :type num_tries: int
        :param validate_function: A function to validate thoughts (if not using LM). Defaults to None.
        :type validate_function: Takes a thought state and returns a boolean.
        """
        super().__init__()
        self.num_samples: int = num_samples
        self.improve: bool = improve
        self.num_tries: int = num_tries
        self.validate_function: Callable[[Dict], bool] = validate_function
        self.thoughts: List[List[Thought]] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the list of final thoughts, after validation and improvement.

        :return: List of final validated and improved thoughts.
        :rtype: List[Thought]
        """
        return [thought_list[-1] for thought_list in self.thoughts]

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the ValidateAndImprove operation by validating and improving the predecessors' thoughts.
        If a validation function is provided, it is used, otherwise the LM is prompted.
        If improvement is enabled, the LM is prompted to improve the thought, if it is not valid.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert (
            len(self.predecessors) > 0
        ), "ValidateAndImprove operation needs at least one predecessor"

        for thought in previous_thoughts:
            thought_list = []
            current_thought = Thought.from_thought(thought)
            current_try = 0
            while True:
                if self.validate_function is not None:
                    self.logger.debug(
                        "Using validate function %s to score states",
                        self.validate_function,
                    )
                    valid = self.validate_function(current_thought.state)
                else:
                    prompt = prompter.validation_prompt(**current_thought.state)
                    self.logger.debug("Prompt for LM: %s", prompt)
                    responses = lm.generate(prompt, num_return_sequences=self.num_samples)
                    self.logger.debug("Responses from LM: %s", responses)

                    valid = parser.parse_validation_answer(
                        current_thought.state, responses
                    )
                current_thought.valid = valid
                thought_list.append(current_thought)
                if (
                    not self.improve
                    or current_thought.valid
                    or current_try >= self.num_tries
                ):
                    break
                improve_prompt = prompter.improve_prompt(**current_thought.state)
                self.logger.debug("Prompt for LM: %s", improve_prompt)
                responses = lm.generate(improve_prompt, num_return_sequences=1)
                self.logger.debug("Responses from LM: %s", responses)
                state_update = parser.parse_improve_answer(
                    current_thought.state, responses
                )
                current_thought = Thought({**current_thought.state, **state_update})
                current_try += 1
            self.thoughts.append(thought_list)

        self.logger.info(
            "Validate and improve operation %d created %d valid thoughts from %d previous thoughts",
            self.id,
            len(
                [
                    thought_list[-1]
                    for thought_list in self.thoughts
                    if thought_list[-1].valid
                ]
            ),
            len(previous_thoughts),
        )


class Generate(Operation):
    """
    Operation to generate thoughts.
    """

    operation_type: OperationType = OperationType.generate

    def __init__(
        self, num_branches_prompt: int = 1, num_branches_response: int = 1
    ) -> None:
        """
        Initializes a new Generate operation.

        :param num_branches_prompt: Number of responses that each prompt should generate (passed to prompter). Defaults to 1.
        :type num_branches_prompt: int
        :param num_branches_response: Number of responses the LM should generate for each prompt. Defaults to 1.
        :type num_branches_response: int
        """
        super().__init__()
        self.num_branches_prompt: int = num_branches_prompt
        self.num_branches_response: int = num_branches_response
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation.

        :return: List of generated thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Generate operation by generating thoughts from the predecessors.
        The thoughts are generated by prompting the LM with the predecessors' thought states.
        If there are no predecessors, the kwargs are used as a base state.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            return

        if len(previous_thoughts) == 0:
            # no predecessors, use kwargs as base state
            previous_thoughts = [Thought(state=kwargs)]

        for thought in previous_thoughts:
            base_state = thought.state
            prompt = prompter.generate_prompt(self.num_branches_prompt, **base_state)
            self.logger.debug("Prompt for LM: %s", prompt)
            responses = lm.generate(prompt, num_return_sequences=self.num_branches_response)
            self.logger.debug("Responses from LM: %s", responses)
            for new_state in parser.parse_generate_answer(base_state, responses):
                new_state = {**base_state, **new_state}
                self.thoughts.append(Thought(new_state))
                self.logger.debug(
                    "New thought %d created with state %s",
                    self.thoughts[-1].id,
                    self.thoughts[-1].state,
                )
        if (
            len(self.thoughts)
            > self.num_branches_prompt
            * self.num_branches_response
            * len(previous_thoughts)
            and self.num_branches_prompt > 0
        ):
            self.logger.warning(
                "Generate operation %d created more thoughts than expected",
                self.id,
            )
        self.logger.info(
            "Generate operation %d created %d new thoughts", self.id, len(self.thoughts)
        )


class Improve(Operation):
    """
    Operation to improve thoughts.
    """

    operation_type: OperationType = OperationType.improve

    def __init__(self) -> None:
        """
        Initializes a new Improve operation.
        """
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation after improvement.

        :return: List of improved thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Improve operation by improving the predecessors' thoughts.
        The thoughts are improved by prompting the LM with the predecessors' thought states.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        assert len(self.predecessors) > 0, "Needs at least one predecessor"

        for thought in previous_thoughts:
            improve_prompt = prompter.improve_prompt(**thought.state)
            self.logger.debug("Prompt for LM: %s", improve_prompt)
            responses = lm.generate(improve_prompt, num_return_sequences=1)
            self.logger.debug("Responses from LM: %s", responses)
            state_update = parser.parse_improve_answer(thought.state, responses)
            self.thoughts.append(Thought({**thought.state, **state_update}))

        self.logger.info(
            "Improve operation %d improved %d thoughts", self.id, len(self.thoughts)
        )


class Aggregate(Operation):
    """
    Operation to aggregate thoughts.
    """

    operation_type: OperationType = OperationType.aggregate

    def __init__(self, num_return_sequences: int = 1) -> None:
        """
        Initializes a new Aggregate operation.

        :param num_return_sequences: Number of responses to use for aggregation. Defaults to 1.
        :type num_return_sequences: int
        """
        super().__init__()
        self.thoughts: List[Thought] = []
        self.num_return_sequences: int = num_return_sequences

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation after aggregation.

        :return: List of aggregated thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Aggregate operation by aggregating the predecessors' thoughts.
        The thoughts are aggregated by prompting the LM with the predecessors' thought states.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        assert (
            len(self.predecessors) >= 1
        ), "Aggregate operation must have at least one predecessor"

        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            return

        # applied in order of score
        base_state: Dict = {}
        for thought in sorted(previous_thoughts, key=lambda thought: thought.score):
            base_state = {**base_state, **thought.state}

        previous_thought_states = [thought.state for thought in previous_thoughts]
        prompt = prompter.aggregation_prompt(previous_thought_states)

        self.logger.debug("Prompt for LM: %s", prompt)

        responses = lm.generate(prompt, num_return_sequences=self.num_return_sequences)

        self.logger.debug("Responses from LM: %s", responses)

        parsed = parser.parse_aggregation_answer(previous_thought_states, responses)

        if isinstance(parsed, dict):
            parsed = [parsed]
        for new_state in parsed:
            self.thoughts.append(Thought({**base_state, **new_state}))


class KeepBestN(Operation):
    """
    Operation to keep the best N thoughts from predecessors based on their score.
    """

    operation_type: OperationType = OperationType.keep_best_n

    def __init__(self, n: int, higher_is_better: bool = True) -> None:
        """
        Initializes a new KeepBestN operation.

        :param n: Maximum number of thoughts to keep.
        :type n: int
        :param higher_is_better: Whether higher scores are better. Defaults to True.
        :type higher_is_better: bool
        :raises AssertionError: If `n` is not greater than zero.
        """
        super().__init__()
        self.n: int = n
        assert self.n > 0, "KeepBestN operation must keep at least one thought"
        self.higher_is_better: bool = higher_is_better
        self.thoughts: List[Thought] = []

    def get_best_n(self) -> List[Thought]:
        """
        Returns the best N thoughts from the predecessors based on their score.

        :return: List of best N thoughts.
        :rtype: List[Thought]
        :raises AssertionError: If not all predecessors have been executed.
        :raises AssertionError: If not all thoughts have been scored.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()
        assert all(
            previous_thought.scored for previous_thought in previous_thoughts
        ), "Not all thoughts have been scored"

        try:
            return sorted(
                previous_thoughts,
                key=lambda thought: thought.score,
                reverse=self.higher_is_better,
            )[: self.n]
        except:
            self.logger.error("Error in KeepBestN operation")
            self.logger.error(
                "Previous operation: %s", [op.id for op in self.predecessors]
            )
            self.logger.error("Previous thoughts: %s", previous_thoughts)
            self.logger.error(
                "Scores: %s", [thought.score for thought in previous_thoughts]
            )
            return sorted(
                [i for i in previous_thoughts if isinstance(i.score, float)],
                key=lambda thought: thought.score,
                reverse=self.higher_is_better,
            )[: self.n]

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts kept by the operation.

        :return: List of kept thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the KeepBestN operation by keeping the best N thoughts from the predecessors according to their score.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        :raises AssertionError: If not all predecessors have been executed.
        :raises AssertionError: If not all thoughts have been scored.
        """
        assert (
            len(self.predecessors) >= 1
        ), "KeepBestN operation must have at least one predecessor"

        self.thoughts = [Thought.from_thought(thought) for thought in self.get_best_n()]

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s kept", thought.id, thought.state
            )

        self.logger.info(
            "KeepBestN operation %d kept %d thoughts", self.id, len(self.thoughts)
        )


class KeepValid(Operation):
    """
    Operation to keep valid thoughts from predecessors.
    """

    operation_type: OperationType = OperationType.keep_valid

    def __init__(self) -> None:
        """
        Initializes a new KeepValid operation.
        """
        super().__init__()
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts kept by the operation.

        :return: List of kept thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the KeepValid operation by keeping the valid thoughts from the predecessors.
        Keeps unvalidated thoughts as well.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessors.
        """
        assert (
            len(self.predecessors) >= 1
        ), "KeepValid operation must have at least one predecessor"

        self.thoughts: List[Thought] = [
            Thought.from_thought(thought)
            for thought in self.get_previous_thoughts()
            if not thought.validated or thought.valid
        ]

        if any(not thought.validated for thought in self.thoughts):
            self.logger.warning(
                "KeepValid operation %d has unvalidated thoughts", self.id
            )

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s kept", thought.id, thought.state
            )

        self.logger.info(
            "KeepValid operation %d kept %d thoughts", self.id, len(self.thoughts)
        )


class GroundTruth(Operation):
    """
    Operation to evaluate if thoughts correctly solve the problem, using a ground truth evaluator
    """

    operation_type: OperationType = OperationType.ground_truth_evaluator

    def __init__(self, ground_truth_evaluator: Callable[[Dict], bool]) -> None:
        """
        Initializes a new GroundTruth operation.

        :param ground_truth_evaluator: A function to evaluate if a thought solves the problem.
        :type ground_truth_evaluator: A function that takes a thought state and returns a boolean.
        """
        super().__init__()
        self.ground_truth_evaluator: Callable[[Dict], bool] = ground_truth_evaluator
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation.

        :return: List of evaluated thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the GroundTruth operation by evaluating the predecessors' thoughts using the ground truth evaluator function.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If operation has no predecessor.
        """
        assert (
            len(self.predecessors) >= 1
        ), "GroundTruth operation must have at least one predecessor"

        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        for thought in previous_thoughts:
            new_thought = Thought.from_thought(thought)
            try:
                new_thought.solved = self.ground_truth_evaluator(new_thought.state)
            except:
                new_thought.solved = False
            self.thoughts.append(new_thought)

        self.logger.info(
            "GroundTruth operation %d evaluated %d thoughts and %d solved the problem",
            self.id,
            len(self.thoughts),
            len([thought for thought in self.thoughts if thought.solved]),
        )


class Selector(Operation):
    """
    Operation to select thoughts from predecessors.
    Useful for separating thoughts to perform different, subsequent operations on them.
    """

    operation_type: OperationType = OperationType.selector

    def __init__(self, selector: Callable[[List[Thought]], List[Thought]]) -> None:
        """
        Initializes a new Selector operation.

        :param selector: A function to select thoughts from the predecessors' thoughts.
        :type selector: A function that takes a list of thoughts and returns a list of thoughts.
        """
        super().__init__()
        self.selector: Callable[[List[Thought]], List[Thought]] = selector
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts selected by the operation.

        :return: List of selected thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, **kwargs
    ) -> None:
        """
        Executes the Selector operation by selecting thoughts from the predecessors using the selector function.
        If the Selector has no predecessors, the selector function is called with a thought containing the kwargs as state.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0:
            previous_thoughts = [Thought(kwargs)]

        self.thoughts = [
            Thought.from_thought(thought)
            for thought in self.selector(previous_thoughts)
        ]

        for thought in self.thoughts:
            self.logger.debug(
                "Thought %d with state %s selected", thought.id, thought.state
            )

        self.logger.info(
            "Selector operation %d selected %d thoughts", self.id, len(self.thoughts)
        )

class Thought:
    """
    Represents an LLM thought with its state, constructed by the parser, and various flags.
    """

    _ids: Iterator[int] = itertools.count(0)

    def __init__(self, state: Optional[Dict] = None) -> None:
        """
        Initializes a new Thought instance with a state and various default flags.

        :param state: The state of the thought. Defaults to None.
        :type state: Optional[Dict]
        """
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Thought._ids)
        self.state: Dict = state
        self._score: float = 0.0
        self._valid: bool = False
        self._solved: bool = False
        self.scored: bool = False
        self.validated: bool = False
        self.compared_to_ground_truth: bool = False

    @staticmethod
    def from_thought(thought: Thought) -> Thought:
        """
        Creates a new thought from an existing one.

        :param thought: An instance of a Thought to clone.
        :return: A new Thought instance with properties copied from the input thought.
        """
        new_thought = Thought(thought.state)
        new_thought.score = thought.score
        new_thought.valid = thought.valid
        new_thought.solved = thought.solved
        new_thought.scored = thought.scored
        new_thought.validated = thought.validated
        new_thought.compared_to_ground_truth = thought.compared_to_ground_truth
        return new_thought

    @property
    def valid(self) -> bool:
        """
        Returns the validity of the thought.

        :return: The validity of the thought.
        :rtype: bool
        """
        return self._valid

    @valid.setter
    def valid(self, valid: bool) -> None:
        """
        Sets the validity of the thought and the validated flag.

        :param valid: The validity of the thought.
        :type valid: bool
        """
        self.validated = True
        self._valid = valid

    @property
    def score(self) -> float:
        """
        Returns the score of the thought.

        :return: The score of the thought.
        :rtype: float
        """
        return self._score

    @score.setter
    def score(self, new_score: float) -> None:
        """
        Sets the score of the thought and the scored flag.

        :param new_score: The score of the thought.
        :type new_score: float
        """
        self.scored = True
        self._score = new_score

    @property
    def solved(self) -> bool:
        """
        Returns the solved flag of the thought.

        :return: The solved flag of the thought.
        :rtype: bool
        """
        return self._solved

    @solved.setter
    def solved(self, solved: bool) -> None:
        """
        Sets the solved flag of the thought and the compared_to_ground_truth flag.

        :param solved: Whether the thought contains a solution to the problem.
        :type solved: bool
        """
        self.compared_to_ground_truth = True
        self._solved = solved
