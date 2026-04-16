from llm_sdk import Small_LLM_Model
import numpy as np


def main():
    model = Small_LLM_Model()

    prompt = """
    You are a function calling system.

    You must choose exactly one function from the list and return ONLY valid JSON.

    Available functions:

    1. fn_add_numbers
    - description: Add two numbers together
    - parameters:
        a: number
        b: number

    2. fn_reverse_string
    - description: Reverse a string
    - parameters:
        s: string

    3. fn_greet
    - description: Greet a person by name
    - parameters:
        name: string

    User request:
    "What is the sum of 2 and 3?"

    Return format (STRICT):
    {
    "name": "...",
    "parameters": { ... }
    }
    """

    for _ in range(100):
        tokens = model.encode(prompt).tolist()[0]
        logits = model.get_logits_from_input_ids(tokens)
        next_token_id = int(np.argmax(logits))
        decoded = model.decode([next_token_id])
        prompt += decoded
    print(prompt)