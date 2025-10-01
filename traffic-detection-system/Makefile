.PHONY: help install install-dev test lint format clean run-api run-demo benchmark docker-build docker-run docker-stop

# Variables
PYTHON := python3
PIP := pip3
PROJECT_NAME := traffic-detection-system
DOCKER_IMAGE := $(PROJECT_NAME):latest
DOCKER_IMAGE_RPI := $(PROJECT_NAME):rpi

# Default target
help:
	@echo "Traffic Detection System - Makefile Commands"
	@echo "============================================"
	@echo "Installation:"
	@echo "  make install        - Install production dependencies"
	@echo "  make install-dev    - Install development dependencies"
	@echo "  make download-models - Download YOLO models"
	@echo ""
	@echo "Development:"
	@echo "  make run-api        - Run API server (development mode)"
	@echo "  make run-demo       - Run live camera demo"
	@echo "  make test          - Run all tests"
	@echo "  make test-unit     - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make benchmark     - Run performance benchmarks"
	@echo "  make lint          - Run code linters"
	@echo "  make format        - Format code with black and isort"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build  - Build Docker image"
	@echo "  make docker-build-rpi - Build Raspberry Pi Docker image"
	@echo "  make docker-run    - Run Docker container"
	@echo "  make docker-stop   - Stop Docker container"
	@echo "  make docker-compose-up - Start all services with docker-compose"
	@echo "  make docker-compose-down - Stop all services"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         - Clean cache and temporary files"
	@echo "  make check-system  - Check system requirements"

# Installation targets
install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-asyncio pytest-cov pytest-mock black isort flake8 mypy

download-models:
	$(PYTHON) main.py download-models

# Development targets
run-api:
	$(PYTHON) main.py serve --reload --env development

run-api-prod:
	$(PYTHON) main.py serve --env production --workers 4

run-demo:
	$(PYTHON) main.py live-demo

process-video:
	@read -p "Enter video path: " video_path; \
	$(PYTHON) main.py process-video $$video_path --show

# Testing targets
test:
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v

benchmark:
	$(PYTHON) benchmarks/benchmark.py

benchmark-quick:
	$(PYTHON) benchmarks/benchmark.py --quick --frames 50

# Code quality targets
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ benchmarks/ main.py --line-length=100
	isort src/ tests/ benchmarks/ main.py --profile=black --line-length=100

check-format:
	black src/ tests/ --check --line-length=100
	isort src/ tests/ --check-only --profile=black --line-length=100

# Docker targets
docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-build-rpi:
	docker build -f Dockerfile.rpi -t $(DOCKER_IMAGE_RPI) .

docker-run:
	docker run -d \
		--name $(PROJECT_NAME) \
		-p 8000:8000 \
		--device /dev/video0:/dev/video0 \
		-v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE)

docker-stop:
	docker stop $(PROJECT_NAME) || true
	docker rm $(PROJECT_NAME) || true

docker-compose-up:
	docker-compose up -d

docker-compose-down:
	docker-compose down

docker-compose-logs:
	docker-compose logs -f

# Utility targets
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true
	rm -rf data/cache/* 2>/dev/null || true
	@echo "Cleanup complete!"

check-system:
	@echo "Checking system requirements..."
	@echo "Python version:"
	@$(PYTHON) --version
	@echo ""
	@echo "Platform:"
	@$(PYTHON) -c "import platform; print(f'System: {platform.system()}'); print(f'Machine: {platform.machine()}')"
	@echo ""
	@echo "OpenCV version:"
	@$(PYTHON) -c "import cv2; print(f'OpenCV: {cv2.__version__}')" 2>/dev/null || echo "OpenCV not installed"
	@echo ""
	@echo "PyTorch availability:"
	@$(PYTHON) -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not installed"
	@echo ""
	@echo "Camera devices:"
	@ls /dev/video* 2>/dev/null || echo "No camera devices found"

# Development shortcuts
dev: install-dev download-models
	@echo "Development environment ready!"

serve: run-api

demo: run-demo

# CI/CD targets
ci-test:
	pytest tests/ -v --cov=src --cov-report=xml

ci-lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503 --format=junit-xml --output-file=lint-report.xml

ci-build:
	docker build -t $(DOCKER_IMAGE) .

# Help is the default target
.DEFAULT_GOAL := help