*This project has been created as part of the 42 curriculum by imelero-.*

# call me maybe

Introduction to function calling in LLMs.

---

## Description

This project implements a lightweight function-calling system using constrained decoding with a small language model.

Instead of allowing the model to generate completely free-form text, the decoding process is restricted token-by-token using logit masking. The model is only allowed to generate tokens that keep the output valid according to predefined function schemas.

The system is capable of:
- Selecting the correct function for a user request.
- Extracting typed parameters from natural language.
- Generating structured JSON outputs.

The project focuses on understanding how modern function-calling systems work internally without relying on external APIs or high-level frameworks.

---

## Features

- Function selection using constrained decoding.
- Parameter extraction for:
  - integers
  - floating-point numbers
  - strings
- Token-level logit masking.
- Schema-aware generation.
- JSON output generation.
- Type-safe Python implementation.
- Static analysis using mypy.
- Code linting using flake8.

---

## Project Structure

```text
.
├── data
│   ├── input
│   │   ├── function_calling_tests.json
│   │   └── functions_definition.json
│   └── output
│       └── function_calls.json
├── src
│   ├── engine.py
│   ├── parser.py
│   └── main.py
├── Makefile
└── README.md
```

## Instructions

### Installation

Install dependencies using:
```bash
make install
```
or manually:
```bash
uv sync
```
### Running the Project

Run the project using:
```bash
make run
```
or:
```bash
uv run python3 -m src
```
### Debug Mode

Run the project using Python's debugger:
```bash
make debug
```

### Cleaning Temporary Files

Remove cache and temporary files:
```bash
make clean
```
### Linting

Run flake8 and mypy checks:
```bash
make lint
```

## Algorithm Explanation

### Function Selection

The project determines the target function using constrained token generation.

At every decoding step:

The model generates logits for the next token.
Invalid tokens are masked with -inf.
Only tokens that can continue a valid function name remain available.
The token with the highest remaining probability is selected.

This guarantees that the generated function name always matches one of the allowed schema definitions.

### Parameter Extraction

Parameters are extracted one token at a time using constrained decoding.

#### The decoder:

Restricts valid tokens depending on parameter type.
Validates partial candidates during generation.
Prevents malformed numbers and invalid continuations.
Rejects tokens not compatible with the current parameter.

For example:

Integers only allow digits and minus signs.
Floating-point numbers only allow a single decimal point.
Strings stop generation when a closing quote is reached.

The implementation also removes already extracted values from the remaining prompt to reduce parameter duplication.

### Logit Masking

The core mechanism of the project is logit masking.

Before selecting the next token:

Invalid tokens are assigned -inf.
Only valid candidates remain selectable.

This transforms unconstrained language generation into deterministic structured generation.

### Design Decisions

Several important design decisions were made during implementation.

#### Token-Level Control

The project avoids full free-form JSON generation because small language models often hallucinate or generate malformed outputs.

Instead, generation is performed incrementally and constrained at every decoding step.

### Schema-Driven Generation

Function schemas define:

parameter names
parameter types
allowed structures

The model uses these schemas as generation constraints rather than relying only on prompting.

### Deterministic Decoding

The project uses argmax decoding instead of sampling.

This improves:

reproducibility
consistency
reliability

while reducing random malformed outputs.

Separate Function and Parameter Extraction

Function selection and parameter extraction are handled independently.

This simplifies:

debugging
validation
token restriction logic

and improves overall system reliability.

### Performance Analysis
Accuracy

The system achieves strong accuracy on:

numeric extraction
simple string extraction
function classification

Accuracy improves significantly when aggressive token masking is applied.

However, string extraction remains more difficult than numeric extraction because language tokens are less predictable and can contain formatting inconsistencies.

#### Speed

The implementation is relatively fast because:

generation is greedy
invalid tokens are masked early
no beam search is used

Most prompts are processed within a small number of decoding steps.

#### Reliability

Reliability is improved through:

schema constraints
type validation
incremental decoding
prompt reduction after extraction

Malformed JSON outputs are largely prevented through constrained generation.

## Challenges Faced

Several important challenges appeared during development.

### Repeated Tokens

The model frequently repeated:

digits
punctuation
decimal points

This caused infinite loops such as:

355555555555...

The issue was mitigated by:

validating candidate values
restricting repeated decimal points
checking prompt consistency

### Incorrect Numeric Extraction

The model sometimes generated:

non-existent decimals
merged numbers
repeated parameters

Example:

23 -> 234567

This happened because partial candidates still matched substrings inside larger numbers.

The solution involved:

removing already extracted values
validating incremental candidates
improving token restrictions

### String Termination

Strings occasionally missed:

closing parentheses
quotation marks
punctuation

Additional stopping conditions and token validation logic were added to improve extraction stability.

## Type Checking with mypy

Strict typing introduced several issues involving:

dynamic dictionaries
mixed variable types
temporary string/number conversions

The codebase was refactored using:

explicit type annotations
temporary typed variables
typed containers
Testing Strategy

## The implementation was tested using multiple categories of prompts.

### Arithmetic Prompt examples

What is the product of 3 and 5?
What is the product of 12 and 4?

### Numeric Prompt examples

Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years

### String Prompt examples

Execute SQL query 'SELECT * FROM users' on the production database
Read C:\Users\john\config.ini with latin-1 encoding

### These tests validated:

floating-point extraction
decimal handling
parameter ordering
SQL Query Extraction
string extraction
quote handling
punctuation preservation
File Path Extraction
escaping
special characters
filesystem paths

### Example Usage

Input:

Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years

Generated output:

{
    "prompt": "Calculate compound interest on 1234567.89 at 0.0375 rate for 23 years",
    "name": "fn_calculate_compound_interest",
    "parameters": {
        "principal": 1234567.89,
        "rate": 0.0375,
        "years": 23
    }
}

## Resources

Documentation
Python documentation:
https://docs.python.org/3/
NumPy documentation:
https://numpy.org/doc/
mypy documentation:
https://mypy.readthedocs.io/
flake8 documentation:
https://flake8.pycqa.org/
Learning Resources
OpenAI function calling overview:
https://platform.openai.com/docs/guides/function-calling
Hugging Face decoding strategies:
https://huggingface.co/docs/transformers/main/en/generation_strategies
Constrained decoding overview:
https://lilianweng.github.io/posts/2021-01-02-controllable-text-generation/

## AI Usage

AI tools were used during development for:

debugging assistance
type annotation suggestions
documentation writing
README generation

The implementation logic, constrained decoding system, token masking strategy, and debugging process were designed and developed manually.
