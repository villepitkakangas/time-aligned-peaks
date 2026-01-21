
# Unified Makefile for Time-Aligned Peaks (TAP)
# Default python interpreter can be overridden: `make PYTHON=python3 demo`
PYTHON ?= python

# Data inputs (override as needed)
PRIMARY ?= examples/synthetic_primary.csv
SECONDARY ?= examples/synthetic_secondary.csv
DATEFMT ?= %Y-%m

# Common CLI flags for time-aligned-peaks
COMMON := --primary $(PRIMARY) \
          --secondary $(SECONDARY) \
          --secondary-date-format "$(DATEFMT)" \
          --title "Synthetic demo" \
          --output-peaks peaks_report.csv \
          --output-peak-matrix peaks_matrix.csv

# Make defaults
.DEFAULT_GOAL := demo

.PHONY: help demo demo_local demo_median demo_min demo_print \
        test lint format coverage clean dist

## Show this help message
help:
    @echo "Available make commands:"
    @echo
    @# Extract targets followed by '##' comments and format them
    @awk 'BEGIN {FS = ":.*## "}; /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-15s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
    @echo
    @echo "Tip: override variables like PRIMARY, SECONDARY, DATEFMT:"
    @echo "  make demo_local PRIMARY=data/primary.csv SECONDARY=data/secondary.csv DATEFMT=%Y-%m"

## Run the default demo (alias of demo_local)
demo: demo_local

## Run demo using local timeline bar width mode (saves demo_local.png)
demo_local:
    $(PYTHON) -m pip install -e . > /dev/null
    time-aligned-peaks $(COMMON) \
      --timeline-bar-width-mode local \
      --timeline-bar-shrink 0.95 \
      --save-figure demo_local.png
    @echo "✅ demo_local complete -> demo_local.png, peaks_report.csv, peaks_matrix.csv"

## Run demo using median timeline bar width mode (saves demo_median.png)
demo_median:
    $(PYTHON) -m pip install -e . > /dev/null
    time-aligned-peaks $(COMMON) \
      --timeline-bar-width-mode median \
      --timeline-bar-shrink 0.90 \
      --save-figure demo_median.png
    @echo "✅ demo_median complete -> demo_median.png, peaks_report.csv, peaks_matrix.csv"

## Run demo using minimum timeline bar width mode (saves demo_min.png)
demo_min:
    $(PYTHON) -m pip install -e . > /dev/null
    time-aligned-peaks $(COMMON) \
      --timeline-bar-width-mode min \
      --timeline-bar-shrink 0.95 \
      --save-figure demo_min.png
    @echo "✅ demo_min complete -> demo_min.png, peaks_report.csv, peaks_matrix.csv"

## Print 'demo' using inline Python (preserves File 2 intent)
demo_print:
    $(PYTHON) -m pip install -e . > /dev/null
    $(PYTHON) -c 'print("demo")'

## Run unit tests with pytest
test:
    $(PYTHON) -m pip install -e . > /dev/null
    $(PYTHON) -m pip install pytest -q
    $(PYTHON) -m pytest -q

## Lint codebase with ruff
lint:
    $(PYTHON) -m pip install ruff -q
    ruff check .

## Format code with black
format:
    $(PYTHON) -m pip install black -q
    black .

## Test with coverage; generate XML, HTML, and text reports
coverage:
    $(PYTHON) -m pip install -e . > /dev/null
    $(PYTHON) -m pip install pytest coverage -q
    coverage run -m pytest -q
    coverage xml
    coverage html
    coverage report -m

## Clean generated artifacts (images, coverage outputs)
clean:
    rm -f demo_local.png demo_median.png demo_min.png peaks_report.csv peaks_matrix.csv
    rm -rf htmlcov .coverage coverage.xml

## Build sdist and wheel using build
dist:
    $(PYTHON) -m pip install build -q
    $(PYTHON) -m build
