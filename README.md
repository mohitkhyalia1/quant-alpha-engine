# Quant Alpha Engine

This repository contains a lightweight quantitative research and backtesting engine used to discover and evaluate trading signals.

## Overview

- `engine.py` — main orchestration script for signal generation and backtests.
- `signal_discovery_v2.py` — signal discovery utilities and scanning routines.
- `validate_and_score.py` — validation and scoring helpers.
- `moccm_grader_modified.py` — grading / evaluation utilities.
- CSV inputs — datasets and rule definitions (e.g. `conditional_rules.csv`, `deterministic_scan.csv`).
- `submissions/` — sample output results.

## Requirements

- Python 3.8+
- Typical dependencies: `pandas`, `numpy`, `scipy`. Install via:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install common packages used by the project:

```bash
pip install pandas numpy scipy
```

## Quickstart

1. Place your input CSV files in the project root (they are already included in this folder).
2. Run the engine:

```bash
python engine.py
```

Inspect output files in the project root or in `submissions/`.

## Notes

- This README is a starting point — update the `Requirements` and `Quickstart` sections with exact package versions and command-line options for `engine.py` if needed.
- If you want, I can add a `requirements.txt`, example config, or command-line usage docs next.

## Contact

Owner: mohitkhyalia1 — https://github.com/mohitkhyalia1/quant-alpha-engine
