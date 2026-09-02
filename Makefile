PYTHON = python3
VENV = venv
UV = uv
PIP = $(VENV)/bin/pip
PYTHON_VENV = $(VENV)/bin/python3
FLAKE8 = $(VENV)/bin/flake8
MYPY = $(VENV)/bin/mypy

.PHONY: install run debug clean lint lint-strict

run:
	$(UV) run $(PYTHON) -m src

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install flake8
	$(PIP) install mypy

debug:
	$(PYTHON) -m pdb a_maze_ing.py

clean:
	rm -rf __pycache__
	rm -rf .mypy_cache
	rm -rf dist
	rm -rf *.egg-info

lint:
	$(FLAKE8) . --exclude $(VENV)
	$(MYPY) --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs .

lint-strict:
	$(FLAKE8) . --exclude $(VENV)
	$(MYPY) --strict .
