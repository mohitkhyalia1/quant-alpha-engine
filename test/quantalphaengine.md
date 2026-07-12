# QuantAlphaEngine

Hackathon submission repository for team **quantalphaengine**.

## Overview

This repository contains the final submission for the **MOCCM 2026 Signal Hunt** competition.

The solution generates both:

* Long-Only portfolio
* Long-Short portfolio

while satisfying all competition constraints and validator requirements.

---

## Repository Structure

```text
.
├── engine.py
├── validate_and_score.py
├── requirements.txt
├── validator_output.txt
├── README.md
└── submissions/
    ├── quantalphaengine_longonly_results.csv
    └── quantalphaengine_longshort_results.csv
```

---

## Requirements

```text
numpy>=2.0
pandas>=2.0
scipy>=1.13
pyarrow>=15.0
numba>=0.60
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Engine

```bash
python engine.py moccm_intraday_blackbox.csv quantalphaengine
```

This generates:

```text
quantalphaengine_longonly_results.csv
quantalphaengine_longshort_results.csv
```

Move them into the submissions folder:

```bash
submissions/
├── quantalphaengine_longonly_results.csv
└── quantalphaengine_longshort_results.csv
```

---

## Validation

Run:

```bash
python validate_and_score.py quantalphaengine --rows 189000
```

Validator status:

```text
RESULT: ALL PASS ✓
```

---

## Final Results

### Long-Only

* Sharpe Ratio: **0.8155**
* Total Return: **17.60%**
* Final Net NAV: **$1,175,978.50**
* Trade Bars: **1,099**
* Total Fees: **$713,484.79**

### Long-Short

* Sharpe Ratio: **0.4177**
* Total Return: **131.93%**
* Final Net NAV: **$2,319,342.15**
* Trade Bars: **26**
* Total Fees: **$3,322.66**

### Blended Sharpe

**0.6166**

---

## Runtime

Measured runtime on the full competition dataset:

```text
~9 seconds
```

which is comfortably below the competition runtime limit.

---

## Team

**Team Name:** quantalphaengine

MOCCM 2026 Signal Hunt Submission
