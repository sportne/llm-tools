VENV ?= $(HOME)/.venvs/llm-tools
PYTHON := $(VENV)/bin/python

SRC_DIR := src
TEST_DIR := tests
PACKAGE := llm_tools
PYTEST_FLAGS ?=

.PHONY: \
	help setup-venv install-dev \
	format format-check \
	lint typecheck dead-code reachability \
	test coverage coverage-report \
	package clean ci

help:
	@echo "llm-tools Makefile targets:"
	@echo "  make setup-venv   - Create shared virtual environment at $(VENV)"
	@echo "  make install-dev  - Install current checkout with dev dependencies"
	@echo "  make format       - Run Ruff formatting"
	@echo "  make format-check - Check Ruff formatting"
	@echo "  make lint         - Run Ruff linting"
	@echo "  make typecheck    - Run mypy static checks"
	@echo "  make dead-code    - Run Vulture dead-code checks"
	@echo "  make reachability - Run mypy and Vulture reachability checks"
	@echo "  make test         - Run the test suite"
	@echo "  make coverage     - Run tests and enforce 90% per-file coverage"
	@echo "  make coverage-report - Run tests with terminal and JSON coverage reporting"
	@echo "  make package      - Build source and wheel distributions"
	@echo "  make clean        - Remove local build and test artifacts"
	@echo "  make ci           - Run format-check, lint, reachability, and coverage gates"

$(PYTHON):
	mkdir -p "$(dir $(VENV))"
	python3 -m venv "$(VENV)"

setup-venv: $(PYTHON)

install-dev: $(PYTHON)
	"$(PYTHON)" -m pip install --upgrade pip
	"$(PYTHON)" -m pip install -e .[dev]

format:
	"$(PYTHON)" -m ruff format $(SRC_DIR) $(TEST_DIR)

format-check:
	"$(PYTHON)" -m ruff format --check $(SRC_DIR) $(TEST_DIR)

lint:
	"$(PYTHON)" -m ruff check $(SRC_DIR) $(TEST_DIR)

typecheck:
	"$(PYTHON)" -m mypy $(SRC_DIR)

dead-code:
	"$(PYTHON)" -m vulture

reachability: typecheck dead-code

test:
	"$(PYTHON)" -m pytest $(PYTEST_FLAGS)

coverage-report:
	"$(PYTHON)" -m pytest $(PYTEST_FLAGS) --cov=$(PACKAGE) --cov-report=term-missing --cov-report=json:coverage.json

coverage: coverage-report
	"$(PYTHON)" scripts/check_coverage.py --input coverage.json --threshold 90

package:
	"$(PYTHON)" -m build --no-isolation

clean:
	rm -rf build dist .coverage .mypy_cache .pytest_cache .ruff_cache *.egg-info src/*.egg-info coverage.json
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

ci: format-check lint reachability coverage
