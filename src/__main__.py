from __future__ import annotations
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError

from pathlib import Path
from typing import Any, List, Dict
from copy import deepcopy
import math
import json
import sys
import os


class Decoder(BaseModel):

    model: Any
    vocab_path: str
    decoding_status: str = "prompt"
    decoding_order: int = 0
    type_restriction: bool = False
    type_order: int = -1

    def next_token(
            self,
            input_ids: List[int],
            parameters: List[str] = []
                ) -> int:

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
            parameters: List[str] = []
                ) -> List[float]:

        token_set_1: List[int] = [90, 330, 40581, 788, 330]
        token_set_2: List[int] = [330, 606, 788, 330]
        token_set_3: List[int] = [330, 13786, 788, 314]

        open_braces_token: List[int] = [90, 314, 341, 515, 972]

        banned_tokens: List[int] = [497, 756, 1153, 2198, 4913, 5212, 57955,
                                    95642]

        # Ban tokens
        for i in banned_tokens:
            logits[i] = -math.inf

        if self.decoding_status == "prompt" and self.decoding_order <= 4:
            i: int = self.decoding_order
            logits[token_set_1[i]] = math.inf

        if self.decoding_status == "prompt" and self.decoding_order > 0:
            for num in open_braces_token:
                logits[num] = -math.inf

        if self.decoding_status == "function" and self.decoding_order <= 3:
            i: int = self.decoding_order
            logits[token_set_2[i]] = math.inf

        if self.decoding_status == "function" and self.decoding_order > 0:
            for num in open_braces_token:
                logits[num] = -math.inf

        if self.decoding_status == "parameter" and self.decoding_order <= 3:
            i: int = self.decoding_order
            logits[token_set_3[i]] = math.inf

        if self.decoding_status == "parameter" and self.decoding_order > 3:
            logits[11693] = -math.inf  # Token Corresponding to \":
            if last_token == 788:
                self.type_restriction = True
                self.type_order += 1
            if last_token == 11 or last_token == 92:
                self.type_restriction = False
            if last_token == 26387:
                logits[3904] = -math.inf    # No _string after regex.
                logits[21260] = -math.inf   # No _pattern after regex.

        number_mode_allowed: List[int] = [i for i in range(15, 25)]
        number_mode_allowed += [11, 92, 220]

        string_mode_prohibited: List[int] = [30, 62, 606, 788, 917, 1355,
                                             1699, 7646]
        string_mode_prohibited += open_braces_token

        if self.type_restriction:
            if parameters[self.type_order] == "number":
                for i, logit in enumerate(logits):
                    if i not in number_mode_allowed:
                        logits[i] = -math.inf

            if parameters[self.type_order] == "string":
                if last_token == 788:       # Token <": >
                    logits[330] = math.inf  # Token < ">
                for i in string_mode_prohibited:
                    logits[i] = -math.inf

            if len(parameters) == self.type_order + 1:
                logits[11] = -math.inf
            else:
                logits[92] = -math.inf

        self.decoding_order += 1
        return logits

    def generate_prompt(
            self,
            functions: List[Dict[str, Any]],
            tests: List[Dict[str, str]],
            func: str,
            sequence: int
                ) -> str:

        prompt: str = ""

        if self.decoding_status == "prompt":
            # request: str = "Copy following prompt in the JSON format.\n"
            request: str = "Copy the following prompt.\n"
            question: Dict[str, str] = tests[sequence]
            question_value: str = question["prompt"]
            question_value = question_value.replace("\"", "\\\"")
            prefix: str = "\"prompt\""

            prompt += request
            prompt += f"{prefix}: \"{question_value}\""

            return prompt

        elif self.decoding_status == "function":

            request: str = "Choose function to solve the prompt.\n"
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
            request: str = "Generate parameters to solve the prompt.\n"
            for d in functions:
                if d["name"] == func:
                    # d.pop("name")
                    d.pop("description")
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
        self.decoding_order = 0

    def get_decoding_status(self) -> str:
        return self.decoding_status

    def reset_decoding_status(self) -> None:
        self.reset_decoding_order()
        self.decoding_status = "prompt"
        self.type_restriction = False
        self.type_order = -1


def read_json_contents(file_path: str) -> Any:

    with open(file_path, 'r') as f:
        contents: str = f.read()
        contents = json.loads(contents)

    return contents


def get_function(generated_json: str) -> str:

    generated: Dict[str, Any] = json.loads(generated_json)
    return generated["name"]


def extract_parameters_type(
        function_list: List[Dict[str, Any]],
        func: str
            ) -> List[str]:

    lst: List[str] = []

    for f in function_list:
        if f["name"] != func:
            continue
        parameters: Dict[str, Any] = f["parameters"]
        for type in parameters.values():
            lst.append(type["type"])

    return lst


def rebuild_preview(
        vocab_dict: Dict[str, int],
        token_set: List[int]
        ) -> str:

    rebuilt_string = ""

    for token in token_set:

        for k, v in vocab_dict.items():
            if v == token:
                k = k.replace('Ġ', ' ')
                k = k.replace('Ċ', '\n')
                rebuilt_string += k

    return rebuilt_string


def print_info(
        vocab_dict: Dict[str, int],
        iteration: int,
        token: int,
        token_set: List[int],
        prompt: str,
        preview: str = ""
        ) -> str:

    for k, v in vocab_dict.items():
        if v == token:
            print(f"{k}, {v}")
            k = k.replace('Ġ', ' ')
            k = k.replace('Ċ', '\n')
            preview += k

    os.system('clear')

    print("===============================")
    print(f"Iteration: {iteration}")
    print()
    print(f"Generated Token: {token_set}")
    print()
    print("Prompt:")
    print(prompt)
    print()
    print("Decode preview:")
    print(preview)
    print()
    print("===============================")

    return preview


def main_loop(
        decoder: Decoder,
        model: Small_LLM_Model,
        function_list: List[Dict[str, Any]],
        calling_tests: List[Dict[str, str]],
        vocab: Dict[str, int],
        id: int,
            ) -> Dict[str, Any]:

    open_braces_token: List[int] = [90, 314, 341, 515, 972]
    close_braces_token: List[int] = [92, 335, 456, 532, 555, 630, 9207]

    decoder.reset_decoding_status()
    prompt: str = decoder.generate_prompt(
                deepcopy(function_list),
                calling_tests,
                None,
                id
            )
    preview: str = ""
    func: str = ""

    encoded_prompt = model.encode(prompt)
    input_ids: List[int] = encoded_prompt[0].tolist()
    output_ids: List[int] = []

    it: int = 0
    braces: int = 0
    while True:
        it += 1
        if decoder.decoding_status == "parameter":
            parameters: List[str] = extract_parameters_type(
                function_list,
                func
            )
            print(parameters)
            token: int = decoder.next_token(input_ids, parameters)
        else:
            token: int = decoder.next_token(input_ids)

        input_ids.append(token)
        output_ids.append(token)

        preview = print_info(
            vocab,
            it,
            token,
            output_ids,
            prompt,
            preview
        )

        if token in open_braces_token:
            braces += 1
        if token in close_braces_token:
            braces -= 1

        if it > 200:
            print("Error: Timed out.", file=sys.stderr)
            break

        if it > 1 and braces == 0:
            if decoder.get_decoding_status() == "parameter":
                break
            else:
                if decoder.decoding_status == "function":
                    func = get_function(preview)
                if output_ids[-1] in close_braces_token:
                    braces += 1
                output_ids[-1] = 345
                preview = rebuild_preview(vocab, output_ids)
                decoder.shift_decoding_status()
                prompt: str = decoder.generate_prompt(
                                    deepcopy(function_list),
                                    calling_tests,
                                    func,
                                    id
                                )
                encoded_prompt = model.encode(prompt)
                input_ids: List[int] = encoded_prompt[0].tolist()
                input_ids += output_ids

    decoded_string: str = model.decode(output_ids)
    try:
        decoded_json: Dict[str, Any] = json.loads(decoded_string)
        return decoded_json
    except json.decoder.JSONDecodeError as e:
        print(f"json.decoder.JSONDecodeError: {e}", file=sys.stderr)
        print(decoded_string, file=sys.stderr)


def main() -> None:

    model = Small_LLM_Model()
    output: List[Dict[str, Any]] = []

    try:
        decoder = Decoder(
            model=model,
            vocab_path=model.get_path_to_vocab_file()
            )
    except ValidationError as e:
        print(e.json())

    script_dir: Path = Path(__file__).resolve().parent
    def_file_path = script_dir.parent / "data" / "input" / "functions_definition.json"
    test_file_path = script_dir.parent / "data" / "input" / "function_calling_tests.json"
    output_path = script_dir.parent / "data" / "output" / "function_calling_result.json"

    vocab_path_local = script_dir.parent / "formatted_vocab.json"

    try:
        function_list: List[Dict[str, Any]] = read_json_contents(def_file_path)
        calling_tests: List[Dict[str, str]] = read_json_contents(test_file_path)

        vocab_dict: Dict[str, int] = read_json_contents(vocab_path_local)
    except Exception as e:
        print(f"FILE MISSING: {e}")
        return

    for i, test in enumerate(calling_tests):
        generated_json: List[Dict[str, Any]] = main_loop(
            decoder,
            model,
            function_list,
            calling_tests,
            vocab_dict,
            i
        )
        output.append(generated_json)

    with open(output_path, 'w') as f:
        f.write(json.dumps(output, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Detected KeyboardInterrupt. Terminating.", file=sys.stderr)
