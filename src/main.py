from llm_sdk.__init__ import Small_LLM_Model
import src.parser as parser
import src.engine as engine
import json
import os


def main() -> None:
    model = Small_LLM_Model()
    try:
        functions_definition, input, output = parser.parse_args()
        functions = parser.parse_functions(functions_definition)
        functions_dict = parser.get_functions_dict(functions)
        function_names = parser.parse_function_names(functions)
        prompts = parser.parse_calls(input)
        parameters = parser.parse_parameters(functions)
    except Exception as e:
        print(f"Error parsing JSON. {e}")
        exit(1)
    data = parser.Structure(output, functions, function_names,
                            prompts, parameters, functions_dict)
    engine.get_json(data, model)
    os.makedirs("data/output", exist_ok=True)
    with open(data.output, "w") as f:
        json.dump(data.result, f, indent=4)
