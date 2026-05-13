import numpy as np
from math import inf
import json


def get_json(data, model):

    for input_prompt in data.prompts:
        print(input_prompt)
        function = get_function_name(data, model, input_prompt)
        print("function done")

        parameters = f"{data.parameters[function][0]}"
        json_str = f'\n{{"prompt":"{input_prompt}","name":"{function}","parameters":{{'

        parameters = get_parameters(data, model, input_prompt, function, json_str)
        print("params done")
        json = {
            "prompt": input_prompt,
            "name": function,
            "parameters": parameters
            }
        data.result.append(json)
        print("done")


def get_jsona(function, parameters, input_prompt, model):
    prompt = (
        "Generate a valid JSON structure with this prompt, function name, and parameters:\n"
        f"Prompt: {input_prompt}.\n"
        f"Function used: {function}.\n"
        f"Parameters used: {parameters}.\n"
    )

    result = "{"
    text_tokens = get_text_tokens(model)
    tokens = model.encode(prompt + result).tolist()[0]

    while '}' not in result:        
        logits = model.get_logits_from_input_ids(tokens)

        for token_id in range(len(logits)):
            if token_id >= len(text_tokens):
                logits[token_id] = float("-inf")
                continue
            str_token = text_tokens[token_id].replace("Ġ", " ")
            str_token = str_token.replace("Ċ", "")

        next_token_id = int(np.argmax(logits))
        tokens.append(next_token_id)
        str_token = text_tokens[next_token_id].replace("Ġ", " ")
        str_token = str_token.replace("Ċ", "")
        result += str_token

    return result

def get_text_tokens(model):
    text_tokens = []
    path = model.get_path_to_vocab_file()

    with open(path, "r") as f:
        tokens_dict = json.load(f)

    text_tokens = [""] * (max(tokens_dict.values()) + 1)

    for text, token_id in tokens_dict.items():
        text_tokens[token_id] = text

    return text_tokens
 
def get_function_name(data, model, input_prompt):
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


def is_valid_function_token(names, token, current):
    for f in names:
        if f.startswith(current + token):
            return True
    return False


def get_parameters(data, model, input_prompt, function, json_str):

    params = data.functions_dict[function]['parameters']
    text_tokens = get_text_tokens(model)
    result = {}
    remaining_prompt = input_prompt

    for p, param_type in params.items():
        prompt = (
            "Extract ONLY the exact parameter value from the user request.\n"
            "Do not invent values.\n"
            "Do not invent decimals.\n"
            "Do not explain.\n"
            "Do not repeat previous parameters.\n"
            "Do not include surrounding text.\n"
            "Copy the exact value from the request.\n\n"

            f"User request: {input_prompt}\n"
            f"Function: {function}\n"
            f"Already extracted parameters: {result}\n"
            f"Parameter name: {p}\n"
            f"Parameter type: {param_type['type']}\n\n"

            f'JSON:\n{json_str}"{p}":'
        )

        tokens = model.encode(prompt).tolist()[0]
        value = ""
        if param_type['type'] == 'integer':
            tokens = model.encode(prompt).tolist()[0]
            while True:
                logits = model.get_logits_from_input_ids(tokens)
                for token_id in range(len(logits)):
                    if token_id >= len(text_tokens):
                        logits[token_id] = float("-inf")
                        continue
                    token = text_tokens[token_id]
                    valid = all(c in "-0123456789" for c in token) or token.strip() in [',', '}']
                    candidate = value + token
                    try:
                        int(candidate)
                    except:
                        valid = False
                    if (not valid or
                    (',' in token and len(value) < 1) or
                    ('}' in token and len(value) < 1) or
                    token not in remaining_prompt or
                    candidate not in remaining_prompt or
                    (token == '0' and value == '0')
                    ):
                        logits[token_id] = float("-inf")

                next_token_id = int(np.argmax(logits))

                token = text_tokens[next_token_id].replace("Ġ", " ").replace("Ċ", "\n")

                if token.strip() in [',', '}'] or np.all(np.isneginf(logits)):
                    print(f"rejected:{token}")
                    break
                tokens.append(next_token_id)
                value += token
                print(value)
            remaining_prompt = remaining_prompt.replace(str(value), "", 1)
            value = int(value.strip())

        if param_type['type'] == 'number':
            while True:
                logits = model.get_logits_from_input_ids(tokens)
                for token_id in range(len(logits)):
                    if token_id >= len(text_tokens):
                        logits[token_id] = float("-inf")
                        continue
                    token = text_tokens[token_id].replace("Ġ", " ")
                    valid = all(c in "-0123456789." for c in token) or token.strip() in [',', '}']
                    candidate = value + token
                    try:
                        float(candidate)
                    except:
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

                token = text_tokens[next_token_id].replace("Ġ", " ").replace("Ċ", "")
                if token.strip() in [',', '}'] or np.all(np.isneginf(logits)):
                    print(f"rejected: {token}")
                    break
                tokens.append(next_token_id)
                value += token
                print(value)
            remaining_prompt = remaining_prompt.replace(str(value), "", 1)
            value = float(value.strip())
        elif param_type['type'] == 'string':
            prompt += '"'
            tokens = model.encode(prompt).tolist()[0]
            while True:
                logits = model.get_logits_from_input_ids(tokens)
                tmp = sorted(enumerate(logits), key = lambda item: item[1], reverse = True )
                i = 0
                for token_id, k in tmp:
                    if token_id >= len(text_tokens):
                        logits[token_id] = float("-inf")
                        continue
                    token = text_tokens[token_id].replace("Ġ", " ").replace("Ċ", "")
                    candidate = value + token
                    if i < 20:
                        #print(token)
                        i += 1
                    if ((token == '"' and len(value) == 0) or
                        (token == "'" and len(value) == 0) or
                        (token == "'" and "'" not in value and '}' not in token) or
                        (token == '"' and '"' not in value and '}' not in token)
                    ):
                        logits[token_id] = float("-inf")
                    if candidate not in input_prompt and '}' not in token:
                        logits[token_id] = float("-inf")
                    
                next_token_id = int(np.argmax(logits))
                tokens.append(next_token_id)
                token = text_tokens[next_token_id].replace("Ġ", " ").replace("Ċ", "\n")
                if '"' in token or np.all(np.isneginf(logits)) or ('}' in token and '{' not in value):
                    print(f"rejected: {token}")
                    break
                value += token
            value = value.strip()
        result[p] = value
    
    return result
