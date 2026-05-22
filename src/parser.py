import json
import sys
from typing import Any


class Structure:
    def __init__(
        self,
        output: str,
        functions: list[dict[str, Any]],
        function_names: list[str],
        prompts: list[str],
        parameters: dict[str, list[str]],
        functions_dict: dict[str, dict[str, Any]]
    ) -> None:
        self.output: str = output
        self.functions: list[dict[str, Any]] = functions
        self.function_names: list[str] = function_names
        self.prompts: list[str] = prompts
        self.parameters: dict[str, list[str]] = parameters
        self.result: list[dict[str, Any]] = []
        self.functions_dict: dict[str, dict[str, Any]] = functions_dict


def parse_args() -> tuple[str, str, str]:
    args = sys.argv[1:]
    functions_definition = "data/input/functions_definition.json"
    input = "data/input/function_calling_tests.json"
    output = "data/output/function_calls.json"

    i = 0
    while i < len(args):
        if args[i] == "--functions_definition":
            functions_definition = args[i + 1]
            i += 2
        elif args[i] == "--input":
            input = args[i + 1]
            i += 2
        elif args[i] == "--output":
            output = args[i + 1]
            i += 2

    return (functions_definition, input, output)


def parse_functions(path: str) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def parse_function_names(
    functions: list[dict[str, Any]]
) -> list[str]:
    function_names = []
    for function in functions:
        function_names.append(function['name'])
    return function_names


def get_functions_dict(
    functions: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    functions_dict = {}
    for function in functions:
        functions_dict[function['name']] = function
    return functions_dict


def parse_parameters(
    functions: list[dict[str, Any]]
) -> dict[str, list[str]]:
    parameters: dict[str, list[str]] = {}

    for function in functions:
        parameters[function['name']] = []
        for param in function['parameters']:
            parameters[function['name']].append(param)

    return parameters


def parse_calls(path: str) -> list[str]:
    with open(path, "r") as f:
        data = json.load(f)

    result = []
    i = 0
    for prompt in data:
        print(f"prompt:{prompt}")
        if prompt['prompt'] == "":
            raise ValueError("Empty prompt.")
        result.append(data[i]['prompt'])
        i += 1
    return result
