from __future__ import annotations
from llm_sdk import Small_LLM_Model

from .decoder import Decoder    # type: ignore

from pydantic import ValidationError
from pathlib import Path
from typing import Any, List, Dict, Union
from copy import deepcopy
import json
import sys
import os


def read_json_contents(file_path: Union[str, Path]) -> Any:

    with open(file_path, 'r') as f:
        contents: str = f.read()
        contents = json.loads(contents)

    return contents


def get_function(generated_json: str) -> str:

    generated: Dict[str, Any] = json.loads(generated_json)
    function_name: str = generated["name"]
    return function_name


def extract_parameters_type(
        function_list: List[Dict[str, Any]],
        func: str
            ) -> Dict[str, str]:

    dct: Dict[str, str] = {}

    for f in function_list:
        if f["name"] != func:
            continue
        parameters: Dict[str, Any] = f["parameters"]
        for param_name, param_info in parameters.items():
            dct[param_name] = param_info["type"]

    return dct


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

    if sys.stdout.isatty():
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


def create_fallback(
        original_prompt: str,
        func_name: str,
        function_list: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
    """Creates a valid fallback JSON object with default types."""
    params: Dict[str, Any] = {}

    for f in function_list:
        if f["name"] == func_name:
            for p_name, info in f["parameters"].items():
                p_type = info["type"]
                if p_type == "number":
                    params[p_name] = 0
                elif p_type == "boolean":
                    params[p_name] = False
                else:
                    params[p_name] = ""
            break

    return {
        "prompt": original_prompt,
        "name": func_name,
        "parameters": params
    }


def validate_and_fix_schema(
        decoded_json: Dict[str, Any],
        original_prompt: str,
        expected_func: str,
        function_list: List[Dict[str, Any]]
            ) -> Dict[str, Any]:

    # 1. Check if top-level keys are exactly what we expect
    required_keys = {"prompt", "name", "parameters"}
    if set(decoded_json.keys()) != required_keys:
        print(
            "Schema Error: Extra or missing top-level keys.",
            file=sys.stderr
            )
        return create_fallback(original_prompt, expected_func, function_list)

    # 2. Check parameter keys match exactly
    expected_params: Dict[str, str] = {}
    for f in function_list:
        if f["name"] == expected_func:
            for param_name, info in f["parameters"].items():
                expected_params[param_name] = info["type"]
            break

    if set(decoded_json["parameters"].keys()) != set(expected_params.keys()):
        print("Schema Error: Missing or extra parameters.", file=sys.stderr)
        return create_fallback(original_prompt, expected_func, function_list)

    # 3. Check parameter types
    for p_name, p_type in expected_params.items():
        val = decoded_json["parameters"][p_name]
        if p_type == "number" and not isinstance(val, (int, float)):
            print(f"Schema Error: {p_name} is not a number.", file=sys.stderr)
            return create_fallback(
                original_prompt,
                expected_func,
                function_list
                )
        if p_type == "string" and not isinstance(val, str):
            print(f"Schema Error: {p_name} is not a string.", file=sys.stderr)
            return create_fallback(
                            original_prompt,
                            expected_func,
                            function_list
                            )
        if p_type == "boolean" and not isinstance(val, bool):
            print(f"Schema Error: {p_name} is not a boolean.", file=sys.stderr)
            return create_fallback(
                           original_prompt,
                           expected_func,
                           function_list
                           )

    # If we reach here, the JSON is 100% valid and schema-compliant!
    return decoded_json


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
                "",
                id
            )
    preview: str = ""
    func: str = ""

    encoded_prompt = model.encode(prompt)
    input_ids: List[int] = encoded_prompt[0].tolist()
    output_ids: List[int] = []

    it: int = 0
    braces: int = 0
    token: int
    while True:
        it += 1
        if decoder.decoding_status == "parameter":
            parameters: Dict[str, str] = extract_parameters_type(
                function_list,
                func
            )
            # Report as incorrect output if Parameters are empty
            if not bool(parameters):
                # Return first defined function as a default.
                create_fallback(
                    calling_tests[id]["prompt"],
                    function_list[0]["name"],
                    function_list
                )
                break
            token = decoder.next_token(input_ids, parameters)
        else:
            token = decoder.next_token(input_ids)

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

        if it > 1000:
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
                prompt = decoder.generate_prompt(
                            deepcopy(function_list),
                            calling_tests,
                            func,
                            id
                        )
                encoded_prompt = model.encode(prompt)
                input_ids = encoded_prompt[0].tolist()
                input_ids += output_ids

    decoded_string: str = model.decode(output_ids)
    original_prompt: str = calling_tests[id]["prompt"]
    try:
        decoded_json: Dict[str, Any] = json.loads(decoded_string, strict=False)
    except json.decoder.JSONDecodeError:
        print("JSON Loading failed. Using absolute fallback.", file=sys.stderr)
        return create_fallback(original_prompt, func, function_list)

    return validate_and_fix_schema(
        decoded_json,
        original_prompt,
        func,
        function_list
    )


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

    def_file: str = "functions_definition.json"
    test_file: str = "function_calling_tests.json"
    output_file: str = "function_calling_results.json"
    for arg1, arg2 in zip(sys.argv[:-1], sys.argv[1:]):
        if arg1 == "--functions_definition":
            def_file = arg2
        if arg1 == "--input":
            test_file = arg2
        if arg1 == "--output":
            output_file = arg2

    data_path: Path = script_dir.parent / "data"
    def_file_path: Path = data_path / "input" / def_file
    test_file_path: Path = data_path / "input" / test_file
    output_path: Path = data_path / "output" / output_file

    vocab_path_local = script_dir.parent / "formatted_vocab.json"

    try:
        function_list: List[Dict[str, Any]] = read_json_contents(def_file_path)
        calling_tests: List[Dict[str, str]] = read_json_contents(
            test_file_path)

        vocab_dict: Dict[str, int] = read_json_contents(vocab_path_local)
    except Exception as e:
        print(f"FILE MISSING: {e}")
        return

    # Complement behaviour upon initializing
    decoder.ban_tokens(vocab_dict)

    for i in range(len(calling_tests)):
        generated_json: Dict[str, Any] = main_loop(
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
    except KeyError:
        print("Detected KeyError.")
        print(
            "Check if passed file is correct. "
            "--functions_definition <definition file>, "
            )
        print("--input <test case file>, --output <output file>")
    except Exception as e:
        print(f"Unexpected Error: {e}")
