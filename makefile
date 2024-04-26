SHELL:=/bin/bash -o pipefail
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON:=python3


all: venv
	source env/bin/activate; python -m pip install --upgrade pip
	source env/bin/activate; pip install -e .

venv:
	if [ ! -d $(ROOT_DIR)/env ]; then $(PYTHON) -m venv $(ROOT_DIR)/env; fi

dev: all
	source env/bin/activate; pip install -e ".[dev]"

black:
	source env/bin/activate; black nomic

isort:
	source env/bin/activate; isort --ignore-whitespace --atomic nomic

pyright:
	source env/bin/activate; pyright nomic/ -p . 

documentation:
	source env/bin/activate; rm -rf /.site && mkdocs build

pypi:
	source env/bin/activate; python setup.py sdist; twine upload dist/*; rm -rf dist/

lint: black isort pyright
	@echo "Lint checks passed!"

pretty: isort black

test:
	source env/bin/activate; pytest -s tests

clean:
	rm -rf {.pytest_cache,env,nomic.egg-info}
	find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf

ci_venv:
	if [ ! -d $(ROOT_DIR)/ci_venv ]; then $(PYTHON) -m venv $(ROOT_DIR)/ci_venv; fi
	source ci_venv/bin/activate; pip install -r ci_venv_requirements.txt

black_ci:
	source env/bin/activate; black --check --diff nomic

isort_ci:
	source env/bin/activate; isort --check --diff --skip nomic

pyright_ci:
	source env/bin/activate; pyright nomic/ -p .
