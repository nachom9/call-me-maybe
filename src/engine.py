import numpy as np
import json
from typing import Any
from src.parser import Structure
from llm_sdk.__init__ import Small_LLM_Model


def get_json(data: Structure, model: Small_LLM_Model) -> None:

    for input_prompt in data.prompts:
        function = get_function_name(data, model, input_prompt)

        json_str = (
            f'\n{{"prompt":"{input_prompt}","name":'
            f'"{function}","parameters":{{'
        )

        parameters = get_parameters(
            data, model, input_prompt, function, json_str
            )
        json = {
            "prompt": input_prompt,
            "name": function,
            "parameters": parameters
            }
        data.result.append(json)


def get_text_tokens(model: Small_LLM_Model) -> list[str]:
    text_tokens = []
    path = model.get_path_to_vocab_file()

    with open(path, "r") as f:
        tokens_dict = json.load(f)

    text_tokens = [""] * (max(tokens_dict.values()) + 1)

    for text, token_id in tokens_dict.items():
        text_tokens[token_id] = text

    return text_tokens


def get_function_name(data: Structure, model: Small_LLM_Model,
                      input_prompt: str) -> Any:
    prompt = (
        "Get the most suitable function name for this user request:\n"
        f"User request: {input_prompt}.\n"
        f"Available functions: {data.functions}.\n"
    )
    result = ""
    names = data.function_names
    text_tokens = get_text_tokens(model)
    tokens = model.encode(prompt).tolist()[0]

    while len(names) > 1:
        logits = model.get_logits_from_input_ids(tokens)

        for token_id in range(len(logits)):
            if token_id >= len(text_tokens):
                logits[token_id] = float("-inf")
                continue
            str_token = text_tokens[token_id].replace("Ġ", " ")
            if not any(f.startswith(result + str_token) for f in names):
                logits[token_id] = float("-inf")

        next_token_id = int(np.argmax(logits))
        tokens.append(next_token_id)
        str_token = text_tokens[next_token_id].replace("Ġ", " ")
        result += str_token
        names = [name for name in names if name.startswith(result)]

    return names[0]


def is_valid_function_token(names: list[str], token: str,
                            current: str) -> bool:
    for f in names:
        if f.startswith(current + token):
            return True
    return False


def get_parameters(data: Structure, model: Small_LLM_Model,
                   input_prompt: str, function: str,
                   json_str: str) -> dict[str, Any]:

    params = data.functions_dict[function]['parameters']
    text_tokens = get_text_tokens(model)
    result: dict[str, Any] = {}
    remaining_prompt = input_prompt
    prompt = (
        "You are a JSON filler.\n"
        "Do not invent values.\n"
        "Do not invent decimals.\n"
        "Do not explain.\n"
        "Do not repeat previous parameters.\n"
        "Do not include surrounding text.\n"
        "Fill this JSON to call this function: "
        f"{data.functions_dict[function]}:\n"

        f'{json_str}'
        )

    for p, param_type in params.items():
        prompt += f'"{p}":'

        tokens = model.encode(prompt).tolist()[0]
        value = ""
        int_value = 0
        float_value = 0.0
        if param_type['type'] == 'integer':
            tokens = model.encode(prompt).tolist()[0]
            while True:
                logits = model.get_logits_from_input_ids(tokens)
                for token_id in range(len(logits)):
                    if token_id >= len(text_tokens):
                        logits[token_id] = float("-inf")
                        continue
                    token = text_tokens[token_id]
                    valid = (
                        all(c in "-0123456789" for c in token) or
                        token.strip() in [',', '}']
                    )
                    candidate = value + token
                    try:
                        int(candidate)
                    except (ValueError, TypeError):
                        valid = False
                    if (
                        not valid or
                        (',' in token and len(value) < 1) or
                        ('}' in token and len(value) < 1) or
                        token not in remaining_prompt or
                        candidate not in remaining_prompt or
                        (token == '0' and value == '0')
                    ):
                        logits[token_id] = float("-inf")

                next_token_id = int(np.argmax(logits))

                token = (
                    text_tokens[next_token_id].
                    replace("Ġ", " ").
                    replace("Ċ", "\n")
                )

                if token.strip() in [',', '}'] or np.all(np.isneginf(logits)):
                    print(f"rejected:{token}")
                    break
                tokens.append(next_token_id)
                value += token
                print(value)
            remaining_prompt = remaining_prompt.replace(str(value), "", 1)
            if not value:
                print(f"Error. No solution for '{input_prompt}'")
            else:
                int_value = int(value.strip())

        if param_type['type'] == 'number':
            while True:
                logits = model.get_logits_from_input_ids(tokens)
                for token_id in range(len(logits)):
                    if token_id >= len(text_tokens):
                        logits[token_id] = float("-inf")
                        continue
                    token = text_tokens[token_id].replace("Ġ", " ")
                    valid = (
                        all(c in "-0123456789." for c in token) or
                        token.strip() in [',', '}']
                    )
                    candidate = value + token
                    try:
                        float(candidate)
                    except (ValueError, TypeError):
                        valid = False
                    if not valid or (
                        ('.' in token and '.' in value) or
                        ('.' in token and len(value) < 1) or
                        candidate not in remaining_prompt or
                        (',' in token and len(value) < 1) or
                        ('}' in token and len(value) < 1) or
                        (token == '0' and value == '0') or
                        ('.' in token and '.' not in remaining_prompt) or
                        '!' in token
                    ):
                        logits[token_id] = float("-inf")

                next_token_id = int(np.argmax(logits))

                token = (
                    text_tokens[next_token_id].
                    replace("Ġ", " ").
                    replace("Ċ", "")
                )
                if token.strip() in [',', '}'] or np.all(np.isneginf(logits)):
                    print(f"rejected: {token}")
                    break
                tokens.append(next_token_id)
                value += token
                print(value)
            remaining_prompt = remaining_prompt.replace(str(value), "", 1)
            if not value:
                print(f"Error. No solution for '{input_prompt}'")
            else:
                float_value = float(value.strip())

        elif param_type['type'] == 'string':
            prompt += '"'
            tokens = model.encode(prompt).tolist()[0]
            while True:
                logits = model.get_logits_from_input_ids(tokens)
                tmp = (
                    sorted(enumerate(logits),
                           key=lambda item: item[1],
                           reverse=True
                           )
                )
                i = 0
                for token_id, k in tmp:
                    if token_id >= len(text_tokens):
                        logits[token_id] = float("-inf")
                        continue
                    token = (
                        text_tokens[token_id].
                        replace("Ġ", " ").
                        replace("Ċ", "")
                    )
                    candidate = value + token
                    if i < 20:
                        i += 1
                    if (
                        (token == '"' and len(value) == 0) or
                        (token == "'" and len(value) == 0) or
                        ('"' in token and token[0] != '"')
                    ):
                        logits[token_id] = float("-inf")

                next_token_id = int(np.argmax(logits))
                tokens.append(next_token_id)
                token = (
                    text_tokens[next_token_id].
                    replace("Ġ", " ").
                    replace("Ċ", "\n")
                )
                if (
                    token[0] == '"' or
                    np.all(np.isneginf(logits)) or
                    (token[0] == '}' and '{' not in value)
                ):
                    break
                value += token
            value = value.strip()
        if param_type['type'] == 'string':
            result[p] = value
        elif param_type['type'] == 'integer':
            result[p] = int_value
        elif param_type['type'] == 'number':
            result[p] = float_value

        if param_type['type'] == 'string':
            prompt += f'{value}",'
        else:
            prompt += f"{value},"

    return result
