"""Constrained decoder that masks LLM logits to enforce a JSON schema.

The decoder keeps a small state machine (prompt, function, parameter) and
modifies the logits returned by the language model before the next token is
selected. Invalid tokens are assigned ``-inf`` and forced tokens are assigned
``+inf``.
"""

from pydantic import BaseModel
from typing import Any, List, Dict
import math
import json
import sys


class Decoder(BaseModel):
    """Stateful controller for token-by-token constrained JSON decoding.

    Attributes:
        model: LLM wrapper used to obtain next-token logits.
        vocab_path: Path to the tokenizer vocabulary file.
        decoding_status: Current stage: ``prompt``, ``function`` or
            ``parameter``.
        decoding_order: Number of tokens generated in the current stage.
        type_restriction: Whether parameter-type masking is active.
        type_order: Index of the parameter currently being generated.
        banned_tokens: Token IDs that may never be generated.
        current_param_name: Name of the parameter currently being generated.
    """

    model: Any
    vocab_path: str
    decoding_status: str = "prompt"
    decoding_order: int = 0
    type_restriction: bool = False
    type_order: int = -1
    banned_tokens: List[int] = [497, 756]
    current_param_name: str = ""

    def ban_tokens(
            self,
            vocab: Dict[str, int]
                ) -> None:
        """Collect token IDs whose text would break the JSON structure.

        Tokens containing a closing brace directly followed by a comma, or
        tokens containing more than one brace, are added to
        :attr:`banned_tokens`. The method is intended to be called once after
        the vocabulary has been loaded.

        Args:
            vocab: Mapping from token text to token ID.
        """
        manually_added: List[int] = [1153, 57955]
        self.banned_tokens += manually_added

        for w in vocab.keys():
            brace_count: int = 0

            if "}," in w:
                self.banned_tokens.append(vocab[w])

            for c in w:
                if c == '{':
                    brace_count += 1
                if c == '}':
                    brace_count += 1

            if brace_count > 1:
                self.banned_tokens.append(vocab[w])

    def next_token(
            self,
            input_ids: List[int],
            parameters: Dict[str, str] = {}
                ) -> int:
        """Select the next token after applying the active constraints.

        Args:
            input_ids: Token IDs that form the current model context.
            parameters: Parameter names mapped to their declared JSON types.

        Returns:
            The ID of the highest-scoring token that survives masking, or
            ``-1`` if no token is left with a finite logit.
        """
        current_token: int = -1
        max_logit: float = -math.inf
        logits: List[float] = self.model.get_logits_from_input_ids(input_ids)
        logits = self.modify_list(logits, input_ids[-1], parameters)

        for i, logit in enumerate(logits):
            if logit > max_logit:
                current_token = i
                max_logit = logit
            if max_logit == math.inf:
                break

        return current_token

    def modify_list(
            self,
            logits: List[float],
            last_token: int,
            parameters: Dict[str, str] = {}
                ) -> List[float]:
        """Apply the structural and type constraints to a logit vector.

        The method bans structurally unsafe tokens, forces the fixed JSON
        scaffolding for the current stage, and restricts parameter values
        according to their declared type. It also advances
        :attr:`decoding_order` and the parameter type state.

        Args:
            logits: Raw logits for every token in the vocabulary.
            last_token: Token selected during the previous generation step.
            parameters: Parameter names mapped to their declared JSON types.

        Returns:
            The modified logit vector. Banned positions are ``-inf`` and
            forced positions are ``+inf``.
        """
        token_set_1: List[int] = [90, 330, 40581, 788, 330]
        token_set_2: List[int] = [330, 606, 788, 330]
        token_set_3: List[int] = [330, 13786, 788, 314]

        open_braces_token: List[int] = [90, 314, 341, 515, 972]

        # General Modifications
        if last_token == 14913:
            logits[92] = math.inf     # } after '"

        # Ban tokens
        for i in self.banned_tokens:
            logits[i] = -math.inf

        if self.decoding_status == "prompt" and self.decoding_order <= 4:
            logits[token_set_1[self.decoding_order]] = math.inf

        if self.decoding_status == "prompt" and self.decoding_order > 0:
            for num in open_braces_token:
                logits[num] = -math.inf

        if self.decoding_status == "function" and self.decoding_order <= 3:
            logits[token_set_2[self.decoding_order]] = math.inf

        if self.decoding_status == "function" and self.decoding_order > 0:
            for num in open_braces_token:
                logits[num] = -math.inf

        if self.decoding_status == "parameter" and self.decoding_order <= 3:
            logits[token_set_3[self.decoding_order]] = math.inf

        if self.decoding_status == "parameter" and self.decoding_order > 3:
            logits[11693] = -math.inf  # Token Corresponding to \":
            if last_token == 788:   # Token <": >
                self.type_restriction = True
                self.type_order += 1
                keys = list(parameters.keys())
                if (self.type_order < len(keys)):
                    self.current_param_name = keys[self.type_order]

            if last_token == 11 or last_token == 92:
                self.type_restriction = False
            if last_token == 26387:
                logits[3904] = -math.inf    # No _string after regex
                logits[21260] = -math.inf   # No _pattern after regex

        number_mode_allowed: List[int] = [i for i in range(15, 25)]
        number_mode_allowed += [11, 92, 220]    # comma, closing brace, space
        number_mode_allowed.append(13)  # decimal point

        boolean_mode_allowed: List[int] = [2514, 4049]  # "True" and "False"
        boolean_mode_allowed += [11, 92, 220]    # comma, closing brace, space

        string_mode_prohibited: List[int] = [30, 62, 606, 788, 917, 1355,
                                             1699, 7646]
        string_mode_prohibited += open_braces_token

        if self.type_restriction and self.current_param_name in parameters:
            p_type: str = parameters[self.current_param_name]

            if p_type == "number":
                for i, logit in enumerate(logits):
                    if i not in number_mode_allowed:
                        logits[i] = -math.inf

            if p_type == "boolean":
                for i, logit in enumerate(logits):
                    if i not in boolean_mode_allowed:
                        logits[i] = -math.inf

            if p_type == "string":
                if last_token == 788:       # Token <": >
                    logits[330] = math.inf  # Token < ">
                for i in string_mode_prohibited:
                    logits[i] = -math.inf

            if len(parameters) == self.type_order + 1:
                logits[11] = -math.inf

        self.decoding_order += 1
        return logits

    def generate_prompt(
            self,
            functions: List[Dict[str, Any]],
            tests: List[Dict[str, str]],
            func: str,
            sequence: int
                ) -> str:
        """Build the natural-language prompt for the current decoding stage.

        In the ``prompt`` stage the prompt asks the model to copy the user
        request. In the ``function`` stage it lists the available function
        names. In the ``parameter`` stage it shows the usage of the selected
        function and asks for its arguments.

        Args:
            functions: Function definitions used to build the prompt. The
                dictionaries are modified in place, so callers should pass a
                deep copy when the originals are still needed.
            tests: Test prompts indexed by ``sequence``.
            func: Function selected in the function stage.
            sequence: Zero-based index of the prompt in ``tests``.

        Returns:
            The prompt text for the current stage.
        """
        prompt: str = ""
        request: str = ""

        if self.decoding_status == "prompt":
            # request: str = "Copy following prompt in the JSON format.\n"
            request = "Copy the following prompt.\n"
            question: Dict[str, str] = tests[sequence]
            question_value: str = question["prompt"]
            question_value = question_value.replace("\"", "\\\"")
            prefix: str = "\"prompt\""

            prompt += request
            prompt += f"{prefix}: \"{question_value}\""

            return prompt

        elif self.decoding_status == "function":

            request = "Choose a function from the list to solve the prompt.\n"
            for d in functions:
                d.pop("description")
                d.pop("parameters")
                d.pop("returns")

            prompt += request
            prompt += json.dumps(functions)

            return prompt

        else:
            has_regex: bool = False

            param_dict: Dict[str, Any] = {}
            request = "Generate parameters to solve the prompt.\n"
            for d in functions:
                if d["name"] == func:
                    d.pop("returns")

                    param_dict = d

            for p in param_dict["parameters"].keys():
                if p == "regex":
                    has_regex = True

            prompt += request
            prompt += "Usage of the function:\n"
            prompt += json.dumps(param_dict)
            if has_regex:
                prompt += "\nUse POSIX regular expression for regex.\n"

            return prompt

    def shift_decoding_status(self) -> None:
        """Advance the decoder to the next generation stage.

        The transition order is ``prompt`` -> ``function`` -> ``parameter``.
        Calling the method from the final stage prints a diagnostic message
        and leaves the state unchanged.
        """
        if self.decoding_status == "prompt":
            self.decoding_status = "function"
            self.reset_decoding_order()
            return
        if self.decoding_status == "function":
            self.decoding_status = "parameter"
            self.reset_decoding_order()
            return
        else:
            print(
                "Something Went Wrong: "
                "You have already reached the last stage.",
                file=sys.stderr
                )
            pass

    def reset_decoding_order(self) -> None:
        """Reset the token counter of the current stage to zero."""
        self.decoding_order = 0

    def get_decoding_status(self) -> str:
        """Return the name of the current decoding stage."""
        return self.decoding_status

    def reset_decoding_status(self) -> None:
        """Reset all decoding state before processing a new prompt."""
