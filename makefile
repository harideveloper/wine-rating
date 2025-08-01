# Variables
VENV_NAME = wine_venv
PYTHON = python3
PIP = $(VENV_NAME)/bin/pip
PYTHON_VENV = $(VENV_NAME)/bin/python
TOOLS = shellcheck shfmt bash
PIPELINE_DIR = pipelines/

.DEFAULT_GOAL := help

.PHONY: all
all: setup install

.PHONY: setup
setup: 
	$(PYTHON) -m venv $(VENV_NAME)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

.PHONY: install-tools
install-tools: 
	brew install $(TOOLS)

.PHONY: install
install: 
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

.PHONY: eda
eda: 
	cd notebooks && jupyter notebook wine_eda_01.ipynb

.PHONY: test-components
test-components:
# 	$(PYTHON_VENV) -m pytest --pyargs $(PIPELINE_DIR)/components/tests -v
	PYTHONPATH=$(PIPELINE_DIR) $(PYTHON_VENV) -m pytest pipelines/components/tests -v

.PHONY: test-shared
test-shared:
# 	$(PYTHON_VENV) -m pytest --pyargs $(PIPELINE_DIR)shared/tests -v
# 	$(PYTHON_VENV) -m pytest $(PIPELINE_DIR)/shared/tests -v
	PYTHONPATH=$(PIPELINE_DIR) $(PYTHON_VENV) -m pytest pipelines/shared/tests -v

.PHONY: e2e-train
e2e-train:
# 	$(PYTHON_VENV) -m pytest --pyargs $(PIPELINE_DIR)/training -v
# 	PYTHONPATH=. python pipelines/run_pipeline.py --pipeline training
	PYTHONPATH=$(PIPELINE_DIR) $(PYTHON_VENV) -m pytest $(PIPELINE_DIR)/training -v

.PHONY: e2e-promote
e2e-promote:
# 	$(PYTHON_VENV) -m pytest --pyargs $(PIPELINE_DIR)/training -v
	PYTHONPATH=$(PIPELINE_DIR) $(PYTHON_VENV) -m pytest $(PIPELINE_DIR)/promotion -v

.PHONY: test-coverage
test-coverage:
	$(PYTHON_VENV) -m pytest $(PIPELINE_DIR) --cov=pipelines --cov-report=html --cov-report=term

.PHONY: train
train:
# 	$(PYTHON_VENV) pipelines/run_training_pipeline.py
# 	python pipelines/run_pipeline.py --pipeline training
	PYTHONPATH=. python pipelines/run_pipeline.py --pipeline training

.PHONY: promote
promote:
# 	$(PYTHON_VENV) pipelines/run_promotion_pipeline.py
# 	python pipelines/run_pipeline.py --pipeline promotion
	PYTHONPATH=. python pipelines/run_pipeline.py --pipeline promotion


.PHONY: test-predict
test-predict: 
	sh test/test_prediction.sh

.PHONY: black
black:
	$(PYTHON_VENV) -m black $(PIPELINE_DIR)

.PHONY: flake
flake:
	$(PYTHON_VENV) -m flake8 --exclude $(VENV_NAME) --max-line-length=180 $(PIPELINE_DIR)

.PHONY: pylint
pylint:
	$(PYTHON_VENV) -m pylint --ignore=$(VENV_NAME) $(PIPELINE_DIR)

.PHONY: shfmt
shfmt:
	find . -name "*.sh" -exec shfmt -w {} \;

.PHONY: shellcheck
shellcheck:
	find . -name "*.sh" -exec shellcheck {} \;

.PHONY: sh-bash
sh-bash:
	find . -name "*.sh" -exec bash -n {} \;

.PHONY: lint-bash
lint-bash: shfmt shellcheck sh-bash
	@echo "All bash linting checks passed"

.PHONY: lint-py
lint-py: black flake pylint
	@echo "All python linting checks passed"

.PHONY: lint
lint-all: lint-bash lint-py
	@echo "All files linting checks passed"

.PHONY: clean
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf pipelines/__pycache__
	rm -rf pipelines/components/__pycache__
	rm -rf pipelines/components/tests/__pycache__
	rm -rf pipelines/online/__pycache__
	rm -rf pipelines/online/tests/__pycache__
	rm -rf test/__pycache__

.PHONY: clean-all
clean-all: clean
	rm -rf $(VENV_NAME)

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all           - Set up the environment and install dependencies"
	@echo "  setup         - Create a virtual environment"
	@echo "  install       - Install dependencies"
	@echo "  install-tools - Install development tools via brew"
	@echo "  eda           - Run the EDA notebook"
	@echo "  run  		   - Run the online prediction pipeline"
	@echo "  test-predict  - Run the sample test prediction testing endpoint"
	@echo "  black         - Format code using black"
	@echo "  flake         - Lint code using flake8"
	@echo "  pylint        - Lint code using pylint"
	@echo "  shfmt         - Format shell scripts using shfmt"
	@echo "  shellcheck    - Lint shell scripts using shellcheck"
	@echo "  sh-bash       - Check shell script syntax using bash"
	@echo "  lint-bash     - Format & lint all bash files"
	@echo "  lint-py       - Format & lint all Python files"
	@echo "  lint          - Format & lint all files"
	@echo "  clean         - Remove Python cache files"
	@echo "  clean-all     - Remove all generated files including virtual environment"
	@echo "  help          - Show this help message"