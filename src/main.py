from llm_sdk import Small_LLM_Model
import src.parser as parser
import src.engine as engine


def main():
    model = Small_LLM_Model()

    functions_definition, input, output = parser.parse_args()
    functions = parser.parse_functions(functions_definition)
    function_names = parser.parse_function_names(functions)
    prompts = parser.parse_calls(input)
    data = parser.Structure(output, functions, function_names, prompts)
    engine.get_json(data, model)

    #print(prompt)

