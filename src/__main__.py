from __future__ import annotations
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError

from pathlib import Path
from typing import Any, List, Dict
import math
import json


class Decoder(BaseModel):

    model: Any
    vocab_path: str

    def next_token(self, input_ids: List[int]) -> int:

        current_token: int = -1
        max_logit: float = -math.inf
        logits: List[float] = self.model.get_logits_from_input_ids(input_ids)

        for i, logit in enumerate(logits):
            if logit > max_logit:
                current_token = i
                max_logit = logit

        return current_token

    def something() -> Any:
        pass


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

    function_list = read_contents(def_file_path)
    calling_tests = read_json_contents(test_file_path)

    prompt: str = "The function list:\n"
    prompt += function_list
    prompt += "\n\n"
    prompt += json.dumps(calling_tests[0])
    prompt += "\n\n"

    prompt += (
        "Choose functions and parameters based on function "
        "list, and generate following JSON Format.\n"
        "\n"
        )

    output_json: Dict[str, Any] = {
        "prompt": "(prompt text)",
        "name": "(function you chose)",
        "parameters": "(parameters for the function)"
    }

    prompt += json.dumps(output_json, indent=4)

    print(prompt)

    encoded_prompt = model.encode(prompt)
    input_ids: List[int] = encoded_prompt[0].tolist()

    it = 0
    for i in range(50):
        token: int = decoder.next_token(input_ids)
        input_ids.append(token)
        it += 1

        if it % 5 == 0:
            print()
            decoded_string: str = model.decode(input_ids)
            print(decoded_string)


if __name__ == "__main__":
    main()
