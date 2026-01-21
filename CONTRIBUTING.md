
# Contributing to Time-Aligned Peaks (TAP)

Thanks for your interest in contributing! TAP is a small, research‑oriented visualization tool
that makes **multi-series peak co-occurrences** easy to spot and **reproducible** (figure + CSV artifacts).
We welcome bug reports, feature requests, documentation improvements, and tests.

This guide explains how to get set up locally, run tests, propose changes, and participate in reviews.

---

## Code of Conduct

By participating, you agree to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

---

## Quick Start (development)

1. **Clone and install in editable mode**
   ```bash
   git clone https://github.com/your-org-or-user/time-aligned-peaks.git
   cd time-aligned-peaks
   python -m pip install -e .[dev]
   ```

If you don’t use extras, install dev tools explicitly: pip install pytest coverage

2. Run tests
   ```bash
   pytest
   ```
See tests/ for unit tests and a CLI smoke test that generates a figure and CSV artifacts.

3. Try the CLI locally

    ```bash
    time-aligned-peaks \
      --primary examples/synthetic_primary.csv \
      --secondary examples/synthetic_secondary.csv \
      --secondary-date-format "%Y-%m" \
      --save-figure demo.png \
      --output-peaks peaks_report.csv \
    ```

## Issue Reporting

* Use GitHub Issues for bugs, feature requests, and questions.
* Before opening a new issue, please search existing ones.
* For bugs, include:
    * OS & Python version, TAP version
    * Minimal reproducible input files or snippets
    * Exact command or code, full traceback, expected vs. actual behavior
    * Screenshots of the figure (if relevant)

## Feature Requests
We value small, well‑documented enhancements that reinforce reproducibility and usability:

* New CLI flags (documented in README)
* Additional export formats (kept simple, CSV preferred)
* Robustness improvements (delimiter sniffing, orientation handling, etc.)
* Tests accompanying new functionality

Please open an issue first to discuss scope and design before coding.

## Pull Requests

1. Fork the repo and create a branch:
    ```bash
    git checkout -b feature/short-descriptorShow more lines
    ```

2. Make changes with clear commits. Keep PRs focused and small.
3. Add/extend tests under tests/ for new behavior.
4. Update README.md and docstrings if you change CLI/API behavior.
5. Ensure pytest passes locally and pre-commit checks (if configured).
6. Push and open a PR. In the PR description:
    * Link the issue (if any)
    * Describe the change, rationale, and any trade-offs
    * Note test coverage additions

## Style & conventions

* Python ≥ 3.9 (or the version listed in README)
* Keep dependencies minimal (numpy, pandas, matplotlib, openpyxl)
* Prefer readable code and docstrings over cleverness

## Testing

* Run all tests:
   ```bash
   pytest
   ```
* To measure coverage (optional):
   ```bash
   coverage run -m pytest && coverage report -m
   ```
**What we test**
* Ingestion (CSV/Excel) & delimiter sniffing
* Orientation auto‑detection/override
* Index alignment & timeline adoption by position
* Peak detection (slope‑change rule)
* CLI smoke (figure + `peaks_report.csv` + `peaks_matrix.csv`)

## Releasing & Archiving (DOI)

1. Update version in package metadata (e.g., pyproject.toml/__version__).
2. Tag and push:
    ```bash
    git tag v0.x.y
	git push origin v0.x.y
    ```
3. Create a Zenodo archive and mint a DOI (used for citation/JOSS).
4. Update README Citation section with the DOI.

## JOSS Checklist (for maintainers)

* OSI-approved LICENSE file (e.g., MIT/Apache‑2.0/BSD‑3‑Clause/0BSD)
* Public repo with browsable source and an Issue tracker
* paper.md + references.bib in the repo (brief software note)
* README with Statement of Need, Install, Examples
* Automated tests & community guidelines (CONTRIBUTING.md, CODE_OF_CONDUCT.md)
* Zenodo DOI after tagging a release

## Acknowledgments

Thanks to everyone who contributes issues, ideas, docs, and tests to make TAP better.
Small, reliable tools with good documentation and artifacts can make everyday research workflows
faster and more trustworthy—your contributions help!

