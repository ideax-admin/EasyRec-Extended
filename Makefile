.PHONY: help install dev test lint format clean

help:
	@echo "EasyRec-Extended Development Commands"
	@echo "======================================"
	@echo "install    - Install package"
	@echo "dev        - Install development dependencies"
	@echo "test       - Run tests"
	@echo "lint       - Run linting"
	@echo "format     - Format code"
	@echo "clean      - Clean build artifacts"

install:
	pip install -r requirements.txt

dev:
	pip install -r requirements.txt
	pip install -e ".[dev]"

test:
	pytest tests/ -v

lint:
	flake8 easyrec_extended tests
	pylint easyrec_extended

format:
	black easyrec_extended tests

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .coverage htmlcov/
