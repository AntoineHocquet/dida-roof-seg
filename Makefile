# Makefile for the task

.PHONY: install test run clean tree viz

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

tree:
	eza --tree --git-ignore --all --level=3   --ignore-glob '.idea|.vscode|dist|build|*.egg-info|venv|.gitignore|.gitkeep'

## variant using 'tree' command
# tree -a -L 3 -I 'venv|.git|__pycache__|.mypy_cache|.pytest_cache|.ruff_cache|.idea|.vscode|node_modules|dist|build|*.egg-info'

viz:
	PYTHONPATH=./src python src/dida_roofseg/viz.py && open outputs/learning_curves.png