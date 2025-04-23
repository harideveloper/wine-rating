# Variables
VENV_NAME = wine_venv
PYTHON = python3
PIP = $(VENV_NAME)/bin/pip
PYTHON_VENV = $(VENV_NAME)/bin/python

# Default target
.PHONY: all
all: venv install

# Create virtual environment
.PHONY: venv
venv:
	$(PYTHON) -m venv $(VENV_NAME)

# Install dependencies
.PHONY: install
install: venv
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

# Run EDA notebook
.PHONY: eda
eda:
	cd notebooks && jupyter notebook wine_rating_eda.ipynb

# Run the pipeline
.PHONY: run-pipeline
run-pipeline:
	cd pipeline && $(PYTHON_VENV) run_pipeline.py

# Test the endpoint
.PHONY: test-endpoint
test-endpoint:
	cd test && $(PYTHON_VENV) test_endpoint.py

# Clean up
.PHONY: clean
clean:
	rm -rf __pycache__
	rm -rf pipeline/__pycache__
	rm -rf test/__pycache__
	rm -rf *.pyc

# Deep clean (remove virtual environment)
.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV_NAME)

# Format code using black
format:
	$(ACTIVATE) && black .

# Lint code using flake8
lint:
	$(ACTIVATE) && flake8 .

# Help
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all           - Set up the environment and install dependencies"
	@echo "  venv          - Create a virtual environment"
	@echo "  install       - Install dependencies"
	@echo "  eda           - Run the EDA notebook"
	@echo "  run-pipeline  - Run the wine rating pipeline"
	@echo "  test-endpoint - Test the deployed model endpoint"
	@echo "  clean         - Remove Python cache files"
	@echo "  clean-all     - Remove all generated files including virtual environment"
	@echo "  init          - Create initial project structure"
	@echo "  format        - black formatter"
	@echo "  lint          - flake8 linter"
	@echo "  help          - Show this help message"