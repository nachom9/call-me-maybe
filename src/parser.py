import json
import sys

class Structure:

    def __init__(self, output, functions, function_names, prompts, parameters, functions_dict):
        self.output = output
        self.functions = functions
        self.function_names = function_names
        self.prompts = prompts
        self.parameters = parameters
        self.result = []
        self.functions_dict = functions_dict

def parse_args():
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


def parse_functions(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def parse_function_names(functions):
    function_names = []
    for function in functions:
        function_names.append(function['name'])
    return function_names


def get_functions_dict(functions):
    functions_dict = {}
    for function in functions:
        functions_dict[function['name']] = function
    return functions_dict


def parse_parameters(functions):
    parameters = {}

    for function in functions:
        parameters[function['name']] = []
        for param in function['parameters']:
            parameters[function['name']].append(param)

    return parameters

def parse_calls(path):
    with open(path, "r") as f:
        data = json.load(f)

    result = []
    i = 0
    for prompt in data:
        result.append(data[i]['prompt'])
        i += 1
    return result