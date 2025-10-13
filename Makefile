# Makefile for the task

.PHONY: install test run clean

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=./src pytest -vv tests/

clean:
	rm -rf */__pycache__ .pytest_cache *.pyc .coverage htmlcov
	rm -rf build dist *.egg-info
	rm -rf .mypy_cache
	rm -rf .ipynb_checkpoints
	rm -rf .pytest_cache
	rm -rf src/*.egg-info
	rm -rf src/dida_roofseg/__pycache__
	

run:
	python -m src.train