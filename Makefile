PYTHON = python3
VENV = .venv
SOURCE = source
UV = uv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python3
FLAKE8 = $(VENV)/bin/flake8
MYPY = $(VENV)/bin/mypy
SRC = src
LLM = llm_sdk

.PHONY: install run debug clean lint lint-strict

run:
	$(UV) run --active $(PYTHON) -m $(SRC)

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install flake8
	$(PIP) install mypy
	$(PIP) install uv

debug:
	$(PYTHON) -m pdb a_maze_ing.py

clean:
	rm -rf $(SRC)/__pycache__
	rm -rf $(LLM)/__pycache__
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf dist
	rm -rf *.egg-info

lint:
	$(FLAKE8) . --exclude=$(VENV),$(LLM)
	$(MYPY) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .

lint-strict:
	$(FLAKE8) . --exclude=$(VENV),$(LLM)
	$(MYPY) --strict .
