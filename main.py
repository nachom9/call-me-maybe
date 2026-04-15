#!/usr/bin/env python3

import llm_sdk as sdk


prompt = "hello"
model = sdk.Small_LLM_Model()
input_ids = model.encode(prompt)
logits = model.get_logits_from_input_ids(input_ids)
best_logit = max(logits)
print(best_logit)
exit(1)
text = model.decode(logits)
print(text)