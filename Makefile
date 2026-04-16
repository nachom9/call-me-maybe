VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

install:
	uv sync

run: install
	uv run python3 -m src

debug:
	uv run python3 -m pdb -m src

clean:
	rm -rf .venv __pycache__ */__pycache__ .pytest_cache uv.lock

re: clean install