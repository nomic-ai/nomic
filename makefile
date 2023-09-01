SHELL:=/bin/bash -o pipefail
ROOT_DIR:=$(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PYTHON:=python3


all: venv
	source env/bin/activate; python -m pip install --upgrade pip
	source env/bin/activate; pip install --use-deprecated=legacy-resolver -e .

venv:
	if [ ! -d $(ROOT_DIR)/env ]; then $(PYTHON) -m venv $(ROOT_DIR)/env; fi

dev: all
	source env/bin/activate; pip install --use-deprecated=legacy-resolver -e .[dev]

black:
	source env/bin/activate; black -l 120 -S --target-version py36 nomic


isort:
	source env/bin/activate; isort  --ignore-whitespace --atomic -w 120 nomic


documentation:
	source env/bin/activate; rm -rf /.site && mkdocs build

pypi:
	source env/bin/activate; python setup.py sdist; twine upload dist/*; rm -rf dist/

lint:
	source env/bin/activate; pylint --rcfile .pylintrc nomic && echo ""
	source env/bin/activate; black --check -l 120 -S --target-version py36 nomic && echo ""
	source env/bin/activate; isort --verbose --ignore-whitespace --atomic -c -w 120 nomic
	@echo "Lint checks passed!"

pretty: isort black

test:
	source env/bin/activate;  pytest -s nomic/tests
clean:
	rm -rf {.pytest_cache,env,nomic.egg-info}
	find . | grep -E "(__pycache__|\.pyc|\.pyo$\)" | xargs rm -rf