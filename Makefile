.PHONY: help install dev test lint format clean proto serve-grpc serve-all

help:
	@echo "EasyRec-Extended Development Commands"
	@echo "======================================"
	@echo "install    - Install package"
	@echo "dev        - Install development dependencies"
	@echo "test       - Run tests"
	@echo "lint       - Run linting"
	@echo "format     - Format code"
	@echo "clean      - Clean build artifacts"
	@echo "proto      - Generate gRPC stubs from protos/recommendation.proto"
	@echo "serve-grpc - Start with gRPC enabled (HTTP + gRPC)"
	@echo "serve-all  - Start with both HTTP and gRPC (alias for serve-grpc)"

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

proto:
	bash protos/generate.sh

serve-grpc:
	GRPC_ENABLED=true GRPC_PORT=50051 python app.py

serve-all:
	GRPC_ENABLED=true GRPC_PORT=50051 python app.py
