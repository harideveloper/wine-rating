# Makefile - Simple version for local development

# Variables
PYTHON := python3
PIP := pip3
PORT := 8080
HOST := 0.0.0.0

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help setup install run test lint

help: ## Show available commands
	@echo "$(GREEN)Available commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-12s$(NC) %s\n", $$1, $$2}'

setup: ## Setup virtual environment
	@echo "$(GREEN)Setting up virtual environment...$(NC)"
	$(PYTHON) -m venv serving-venv
	@echo "$(YELLOW)Activate with: source serving-venv/bin/activate$(NC)"
	@echo "$(YELLOW)Then run: make install$(NC)"

install: ## Install dependencies
	@echo "$(GREEN)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt

run: ## Run the server locally
	@echo "$(GREEN)Starting server...$(NC)"
	@if [ -f .env ]; then \
		set -a && source .env && set +a && uvicorn main:app --host $(HOST) --port $(PORT) --reload; \
	else \
		echo "$(YELLOW)No .env file found, running without GCS access$(NC)"; \
		uvicorn main:app --host $(HOST) --port $(PORT) --reload; \
	fi

test: ## Run tests
	@echo "$(GREEN)Running tests...$(NC)"
	pytest tests/ -v

lint: ## Run code linting
	@echo "$(GREEN)Running linting...$(NC)"
	flake8 main.py tests/ --max-line-length=100 --ignore=E203,W503
	black --check main.py tests/