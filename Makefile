VENV=.venv
PYTHON=$(VENV)/bin/python
PIP=$(VENV)/bin/pip

install:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

run:
	$(PYTHON) main.py

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache

re:
	make clean
	make install