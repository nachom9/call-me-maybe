import json
import sys
from typing import Any


class Structure:
    """
    Stores all parsed project data required during execution.

    This structure centralizes:
    - Function schemas.
    - Function names.
    - Input prompts.
    - Parameter definitions.
    - Generated results.

    Attributes:
        output: Output JSON file path.
        functions: List containing all function definitions.
        function_names: List of available function names.
        prompts: List of user prompts to process.
        parameters: Mapping between function names and parameter names.
        result: Final generated function-calling results.
        functions_dict: Dictionary mapping function names to their
            complete schema definitions.
    """
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
    """
    Parses command-line arguments.

    Supported arguments:
    - --functions_definition
    - --input
    - --output

    If an argument is not provided, a default path is used.

    Returns:
        A tuple containing:
        - Functions definition file path.
        - Input prompts file path.
        - Output file path.
    """

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
    """
    Loads the function definitions JSON file.

    Args:
        path: Path to the JSON file containing function schemas.

    Returns:
        A list of dictionaries representing the function definitions.
    """
    with open(path, "r") as f:
        data: list[dict[str, Any]] = json.load(f)
    return data


def parse_function_names(
    functions: list[dict[str, Any]]
) -> list[str]:
    """
    Extracts all function names from the function definitions.

    Args:
        functions: List of function schema dictionaries.

    Returns:
        A list containing all available function names.
    """
    function_names = []
    for function in functions:
        function_names.append(function['name'])
    return function_names


def get_functions_dict(
    functions: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """
    Creates a dictionary mapping function names to their schemas.

    Args:
        functions: List of function schema dictionaries.

    Returns:
        A dictionary where:
        - Keys are function names.
        - Values are full function definition dictionaries.
    """

    functions_dict = {}
    for function in functions:
        functions_dict[function['name']] = function
    return functions_dict


def parse_parameters(
    functions: list[dict[str, Any]]
) -> dict[str, list[str]]:
    """
    Extracts parameter names for each function.

    Args:
        functions: List of function schema dictionaries.

    Returns:
        A dictionary where:
        - Keys are function names.
        - Values are lists containing the parameter names required
          by each function.
    """

    parameters: dict[str, list[str]] = {}

    for function in functions:
        parameters[function['name']] = []
        for param in function['parameters']:
            parameters[function['name']].append(param)

    return parameters


def parse_calls(path: str) -> list[str]:
    """
    Loads and validates the input prompts file.

    Ensures that all prompts are non-empty before returning them.

    Args:
        path: Path to the JSON file containing user prompts.

    Returns:
        A list containing all prompt strings.

    Raises:
        ValueError: If an empty prompt is found.
    """

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
