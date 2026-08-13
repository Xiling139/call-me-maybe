from __future__ import annotations
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError

from pathlib import Path
from typing import Any, List, Dict
import math
import json
import sys


class Decoder(BaseModel):

    model: Any
    vocab_path: str
    decoding_state: int = 0

    def next_token(self, input_ids: List[int]) -> int:

        current_token: int = -1
        max_logit: float = -math.inf
        logits: List[float] = self.model.get_logits_from_input_ids(input_ids)
        logits = self.modify_list(logits)

        for i, logit in enumerate(logits):
            if logit > max_logit:
                current_token = i
                max_logit = logit
            if max_logit == math.inf:
                break

        return current_token

    def modify_list(self, logits: List[float]) -> List[float]:

        if self.decoding_state == 0:
            logits[90] = math.inf

        self.decoding_state += 1
        return logits

    def reset_decoding_state(self) -> None:
        self.decoding_state = 0


def read_contents(file_path: str) -> str:

    with open(file_path, 'r') as f:
        contents: str = f.read()
    return contents


def read_json_contents(file_path: str) -> Any:

    with open(file_path, 'r') as f:
        contents: str = f.read()
        contents = json.loads(contents)

    return contents


def main() -> None:

    model = Small_LLM_Model()
    output: List[Dict[str, Any]] = []

    open_braces_token: List[int] = [90, 314, 341, 515, 972]
    close_braces_token: List[int] = [92, 335, 456, 532, 555, 630]

    try:
        decoder = Decoder(
            model=model,
            vocab_path=model.get_path_to_vocab_file()
            )
    except ValidationError as e:
        print("Explain ")
        print(e.json())

    script_dir: Path = Path(__file__).resolve().parent
    def_file_path = script_dir.parent / "data" / "input" / "functions_definition.json"
    test_file_path = script_dir.parent / "data" / "input" / "function_calling_tests.json"
    output_path = script_dir.parent / "data" / "output" / "function_calls.json"

    function_list = read_contents(def_file_path)
    calling_tests = read_json_contents(test_file_path)

    for i, test in enumerate(calling_tests):

        print(f"\n=== [DEBUG {i}/{len(calling_tests)}] ===")

        decoder.reset_decoding_state()

        prompt: str = "The function list:\n"
        prompt += function_list
        prompt += "\n"

        prompt += (
            "Choose a function and its parameters to solve the question: \n"
            f"[{test['prompt']}]\n"
            "based on the function list, and generate results in the "
            "following JSON Format.\n"
            "\n"
            )

        output_json: Dict[str, Any] = {
            "prompt": "",
            "name": "",
            "parameters": {}
        }

        prompt += json.dumps(output_json, indent=4)
        prompt += "\n"

        encoded_prompt = model.encode(prompt)
        input_ids: List[int] = encoded_prompt[0].tolist()
        output_ids: List[int] = []

        it: int = 0
        braces: int = 0
        while True:
            it += 1
            token: int = decoder.next_token(input_ids)
            input_ids.append(token)
            output_ids.append(token)

            print(f"Iterations: {it}, Token: {token}")

            if token in open_braces_token:
                braces += 1
            if token in close_braces_token:
                braces -= 1

            if it > 10 and braces == 0:
                break

        decoded_string: str = model.decode(output_ids)
        try:
            decoded_json: Dict[str, Any] = json.loads(decoded_string)
            output.append(decoded_json)
        except json.decoder.JSONDecodeError as e:
            print(f"json.decoder.JSONDecodeError: {e}", file=sys.stderr)
            print(decoded_string, file=sys.stderr)

    with open(output_path, 'w') as f:
        f.write(json.dumps(output, indent=4))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Detected KeyboardInterrupt. Terminating.", file=sys.stderr)
