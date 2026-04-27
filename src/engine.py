import numpy as np
from math import inf
import json


def get_json(data, model):

    for input_prompt in data.prompts:
        function = get_function_name(data, model, input_prompt)
        #parameters = get_parameters(data, model, input_prompt)
        print(function)


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
        f"User request: {input_prompt}\n"
        f"Available functions: {data.functions}"
    )
    print(input_prompt)
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


def get_parameters(data, model, input_prompt):
    prompt = (
        "Get the parameters of this user request:\n"
        f"User request: {input_prompt}\n"
        f"Available functions: {data.functions}"
    )
    result = ""

    for _ in range(50):
        tokens = model.encode(prompt).tolist()[0]
        logits = model.get_logits_from_input_ids(tokens)
        next_token_id = int(np.argmax(logits))
        decoded = model.decode([next_token_id])
        prompt += decoded
        result += decoded
    
    return result