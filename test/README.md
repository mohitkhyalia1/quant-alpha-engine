# ⚡ Quant Alpha Engine

> **Black-Box Signal Hunt — MOCCM IITB 2026**
> A high-performance quantitative trading research framework for statistical alpha discovery, signal generation, and large-scale backtesting on multi-million-row financial datasets — **zero machine learning**.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [1. Generate Sample Data](#1-generate-sample-data)
  - [2. Run the Full Pipeline](#2-run-the-full-pipeline)
  - [3. Parameter Tuning](#3-parameter-tuning)
  - [4. Hardware Profiling](#4-hardware-profiling)
- [Engine Modules](#engine-modules)
  - [Alpha Discovery (`discovery.py`)](#alpha-discovery-discoverypy)
  - [Signal Generator (`signal.py`)](#signal-generator-signalpy)
  - [Backtester & Risk Manager (`backtest.py`)](#backtester--risk-manager-backtestpy)
  - [Data Loader (`loader.py`)](#data-loader-loaderpy)
  - [Reporter (`report.py`)](#reporter-reportpy)
- [Signal Logic Deep Dive](#signal-logic-deep-dive)
- [Risk Management Rules](#risk-management-rules)
- [Output Files](#output-files)
- [CLI Reference](#cli-reference)
- [Performance Benchmarks](#performance-benchmarks)
- [Competition Compliance](#competition-compliance)
- [Requirements](#requirements)

---

## Overview

**Quant Alpha Engine** is a fully self-contained research system built for the MOCCM IITB 2026 *Black-Box Signal Hunt* competition. Given a CSV with ~50 unlabeled numeric columns, the engine must:

1. **Discover** which column is the tradable asset (`TICKER_00`) using 7 pure-statistics tests — no labels, no hints.
2. **Generate** BUY / SELL / HOLD signals using a dual-EMA crossover combined with ROC momentum and z-score mean reversion.
3. **Backtest** with realistic transaction costs (commission + slippage), enforce all competition risk rules, and compute standard performance metrics.
4. **Write** all required submission files to `outputs/` — all within a 30-second wall-clock budget.

Everything runs on **pure NumPy + Python stdlib**. No PyTorch, TensorFlow, scikit-learn, or any ML library.

---

## Key Features

- **Automatic ticker discovery** — 7-test composite scoring identifies the price series without any column labels
- **Vectorised signal computation** — O(N) rolling statistics via cumsum trick; no inner Python loops for statistics
- **Event-driven backtester** — per-trade state machine with realistic cost modelling (0.1% commission + 0.05% slippage)
- **Walk-forward parameter tuning** — grid search with out-of-sample fold validation; saves best params automatically
- **Hard risk guardrails** — 20% max position size, cash-floor DQ detection, short-selling support
- **Zero-dependency** — requires only `numpy>=1.24`; runs anywhere Python ≥ 3.8 is installed
- **30-second budget compliance** — wall-clock timer with warning if limit is breached
- **Full output suite** — metrics CSV, trade log, signal series, equity curve, alpha scores

---

## Architecture

```
CSV Dataset
    │
    ▼
┌──────────────┐
│  DataLoader  │  ← Ingests multi-column CSV, extracts numeric columns
└──────┬───────┘
       │
       ▼
┌────────────────┐
│ AlphaDiscovery │  ← Scores all columns with 7 statistical tests
│                │    → Returns best candidate column (TICKER_00)
└──────┬─────────┘
       │
       ▼  price series
┌─────────────────┐
│ SignalGenerator │  ← Dual EMA + ROC momentum + Z-score mean reversion
│                 │    → Returns int8 signal array (+1 / 0 / -1)
└──────┬──────────┘
       │
       ▼
┌──────────────────────────┐
│  Backtester + RiskManager│  ← Event-driven simulation with cost model
│                          │    → Equity curve, trade log, metrics dict
└──────┬───────────────────┘
       │
       ▼
┌──────────┐
│ Reporter │  ← Prints metrics, writes all output CSVs
└──────────┘
```

---

## Repository Structure

```
quant-alpha-engine/
│
├── run.py                    ← Competition entry point (judges run this)
├── parameter_tuner.py        ← Grid-search + walk-forward param optimiser
├── benchmark.py              ← Hardware profiler (measures wall-clock time)
├── generate_sample_data.py   ← Synthetic test data generator
├── requirements.txt          ← numpy>=1.24
├── __init__.py
│
├── engine/
│   ├── loader.py             ← CSV ingestion → numpy array
│   ├── discovery.py          ← Identifies TICKER_00 from 50 columns
│   ├── signal.py             ← BUY / SELL / HOLD signal generation
│   ├── backtest.py           ← Event-driven backtester + RiskManager
│   └── report.py             ← Metrics printer
│
├── data/                     ← Place competition CSV here
├── outputs/                  ← All results are written here
└── docs/
    └── methodology.md        ← Mathematical derivations
```

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mohitkhyalia1/quant-alpha-engine.git
cd quant-alpha-engine

# 2. Install the only dependency
pip install -r requirements.txt

# 3. Generate synthetic test data
python generate_sample_data.py

# 4. Run the full pipeline
python run.py --data data/sample_dataset.csv
```

Results are written to `outputs/` immediately.

---

## Detailed Usage

### 1. Generate Sample Data

```bash
python generate_sample_data.py
```

Creates `data/sample_dataset.csv` — a synthetic multi-million-row dataset with ~50 columns, one of which mimics a real price series with autocorrelation and volatility clustering. Use this to validate the pipeline and tune parameters before competition day.

### 2. Run the Full Pipeline

```bash
# Minimal — auto-discovers ticker column
python run.py --data data/competition_data.csv

# With pre-tuned parameters (recommended)
python run.py --data data/competition_data.csv --fast 20 --slow 60

# Force a known column name (skips auto-discovery, saves time)
python run.py --data data/competition_data.csv --ticker TICKER_00

# Full parameter control
python run.py --data data/competition_data.csv \
              --fast 20 --slow 60 \
              --zscore-entry 2.0 --zscore-window 100
```

### 3. Parameter Tuning

Run this **before competition day** on your sample data:

```bash
python parameter_tuner.py --data data/sample_dataset.csv --ticker TICKER_00
```

This performs a grid search across 192 parameter combinations (fast × slow × zscore_entry × zscore_window), evaluated with 3-fold walk-forward validation. Results are ranked by mean out-of-sample Sharpe ratio and the best combination is saved to `outputs/best_params.txt`.

**Parameter grid searched:**

| Parameter       | Values tested               |
|-----------------|-----------------------------|
| `--fast`        | 5, 10, 20, 30               |
| `--slow`        | 30, 60, 100, 150            |
| `--zscore-entry`| 1.5, 2.0, 2.5, 3.0         |
| `--zscore-window`| 50, 100, 200               |

### 4. Hardware Profiling

```bash
python benchmark.py --data data/sample_dataset.csv --ticker TICKER_00
```

Reports wall-clock time for each pipeline stage (loading, discovery, signal generation, backtesting) on your specific hardware. Use this to ensure you stay within the 30-second competition budget.

---

## Engine Modules

### Alpha Discovery (`discovery.py`)

`AlphaDiscovery` scores every numeric column using **7 vectorised statistical tests** and returns the column with the highest composite score as `TICKER_00`. All tests are pure NumPy — no Python row loops.

| Test | Weight | What it detects |
|------|--------|-----------------|
| Serial Autocorrelation (lag 1 & 5) | 2.0 | Price series memory / trend persistence |
| Variance Ratio (k=5) | 1.5 | Deviation from random walk |
| ADF Proxy (OLS t-statistic) | 1.0 | Non-stationarity fingerprint |
| Return Distribution (skew + kurtosis) | 1.0 | Fat-tailed / asymmetric return profile |
| Lead-Lag Cross-Correlation (lags 1,5,10,20) | 2.5 | Strongest discriminator across column pairs |
| Volatility Clustering / ARCH effect | 2.0 | Squared-return autocorrelation |
| Temporal Stability | 1.5 | Correlation consistency across time chunks |

**Composite score formula:**
```
score = 2.0·AC + 1.5·VR + 1.0·ADF + 1.0·Dist + 2.5·LL + 2.0·ARCH + 1.5·Stab
```

To avoid O(C²) cross-column comparisons, lead-lag and stability tests sample a random subset of up to 15 columns.

### Signal Generator (`signal.py`)

Generates a bar-by-bar integer signal array of `+1` (BUY), `0` (HOLD), or `-1` (SELL).

**Strategy — three components fused:**

**1. Dual EMA Crossover (trend)**
```
EMA(t) = α · price(t) + (1 − α) · EMA(t−1),   α = 2 / (window + 1)
cross  = sign(EMA_fast − EMA_slow)
```
Defaults: `FAST_WINDOW = 20`, `SLOW_WINDOW = 60`

**2. Rate-of-Change Momentum confirmation**
```
ROC(t) = (price(t) − price(t − 10)) / price(t − 10)
mom    = sign(ROC)
```
Trend signal is only active when EMA crossover and ROC momentum **agree**:
```
trend = cross   if cross == mom  else  0
```

**3. Z-score Mean Reversion override**
```
z(t) = (price(t) − μ_w(t)) / σ_w(t)      [rolling window w = 100]
mr   = −1  if z > +2σ   (overbought → sell)
mr   = +1  if z < −2σ   (oversold  → buy)
```
At price extremes, mean-reversion overrides the trend signal:
```
final[|z| > threshold] = mr[|z| > threshold]
```

Rolling mean and standard deviation are computed using an **O(N) cumsum trick** (no inner loops):
```
Var = E[X²] − E[X]²   (clipped to ≥ 0)
```

**Performance:** EMA computation benchmarks at ~0.3 s for 9.4 M rows on modern hardware.

### Backtester & Risk Manager (`backtest.py`)

An **event-driven simulation** that iterates bar-by-bar and executes trades when the signal flips. A scalar event loop is used because order state (cash, position, entry price) must be updated per event — identical architecture to production engines like Zipline or Backtrader.

**`RiskManager` — competition rules enforced:**

| Rule | Value |
|------|-------|
| Initial capital | $1,000,000 |
| Max position size | 20% of current equity |
| Commission per trade | 0.10% |
| Slippage per trade | 0.05% |
| Cash floor | $0 (breach = immediate DQ) |

**Trade types handled:** BUY (open long), SELL (close long), SHORT (open short), COVER (close short).

**Metrics computed:**

| Metric | Description |
|--------|-------------|
| `total_return_pct` | (Final equity − Initial) / Initial × 100 |
| `sharpe_ratio` | Mean daily return / StdDev × √252 |
| `sortino_ratio` | Mean daily return / Downside StdDev × √252 |
| `max_drawdown_pct` | Peak-to-trough equity decline |
| `win_rate_pct` | % of closed trades with positive PnL |
| `avg_win` / `avg_loss` | Mean PnL of winning / losing trades |
| `num_trades` | Total trade events |
| `sim_time_sec` | Wall-clock simulation time |

### Data Loader (`loader.py`)

Reads the competition CSV into a 2D NumPy array, identifies all numeric columns, and exposes `get_column(name)` for named price extraction. Forward-fill is applied to NaN/Inf values before any signal computation.

### Reporter (`report.py`)

Prints a formatted metrics summary to stdout during the run. All file I/O is handled in `run.py`'s `_save_all()` function.

---

## Signal Logic Deep Dive

```
Bar t:
│
├── EMA_fast(t)  = α_f · p(t) + (1−α_f) · EMA_fast(t−1)
├── EMA_slow(t)  = α_s · p(t) + (1−α_s) · EMA_slow(t−1)
│
├── cross(t)     = +1 if EMA_fast > EMA_slow, else -1
│
├── ROC(t)       = (p(t) − p(t−10)) / p(t−10)
├── mom(t)       = +1 if ROC > 0, else -1
│
├── trend(t)     = cross(t)  if cross(t) == mom(t)
│                          0  otherwise
│
├── z(t)         = (p(t) − rolling_mean_w(t)) / rolling_std_w(t)
│
└── signal(t)    = mean_reversion(t)   if |z(t)| > zscore_entry
                 = trend(t)            otherwise
```

The hybrid design means the strategy **follows trends** in normal market conditions and **fades extremes** during large deviations — two complementary return regimes.

---

## Risk Management Rules

All rules are enforced inside `RiskManager` and `Backtester.run()`:

```
Cash Balance must NEVER go below 0
  → If cash < 0 at any bar: simulation halts, disqualified = True

Position sizing:
  shares = (cash × 0.20) / (price × (1 + slippage) × (1 + commission))

Effective fill price:
  Buy  fill = price × (1 + 0.0005)    [slippage away from mid]
  Sell fill = price × (1 − 0.0005)
  Commission deducted from cash on both sides at 0.001 × trade_value
```

---

## Output Files

All files are written to `outputs/` after a successful run:

| File | Description |
|------|-------------|
| `submission_metrics.csv` | Sharpe, Sortino, PnL, drawdown, win rate, trade count, DQ status |
| `trades.csv` | Every trade: bar index, action (BUY/SELL/SHORT/COVER), price, shares, PnL, cash |
| `signals.csv` | Bar-by-bar signal array: `+1`, `0`, or `-1` |
| `equity_curve.csv` | Portfolio equity at every bar |
| `alpha_scores.csv` | All columns ranked by composite alpha score (shows discovery logic) |
| `best_params.txt` | Best EMA/z-score params from parameter tuner |

---

## CLI Reference

### `run.py` — Main Entry Point

```
python run.py --data <path> [options]

Required:
  --data PATH           Path to competition CSV dataset

Optional:
  --ticker NAME         Force column name (skips auto-discovery)
  --fast INT            EMA fast window  (default: 20)
  --slow INT            EMA slow window  (default: 60)
  --zscore-entry FLOAT  Z-score threshold for mean reversion (default: 2.0)
  --zscore-window INT   Rolling window for z-score (default: 100)
```

### `parameter_tuner.py`

```
python parameter_tuner.py --data <path> --ticker <name> [--splits N]

  --data PATH     Dataset for grid search
  --ticker NAME   Column to backtest on
  --splits INT    Walk-forward fold count (default: 3)
```

### `benchmark.py`

```
python benchmark.py --data <path> --ticker <name>
```

### `generate_sample_data.py`

```
python generate_sample_data.py
# Writes data/sample_dataset.csv (no arguments needed)
```

---

## Performance Benchmarks

The engine is designed to complete the full pipeline within 30 seconds on typical competition hardware:

| Stage | Typical Time |
|-------|-------------|
| CSV loading | < 2 s |
| Alpha discovery (50 columns) | 3 – 8 s |
| Signal generation (9.4 M rows) | ~0.3 s |
| Event-driven backtest | < 2 s |
| File I/O | < 1 s |
| **Total** | **< 15 s** |

> Use `benchmark.py` to profile your specific hardware before competition day. A warning is printed if wall time exceeds 30 s.

---

## Competition Compliance

| Requirement | Status |
|-------------|--------|
| No PyTorch / TensorFlow / scikit-learn | ✅ |
| No Neural Networks, Random Forests, or LLMs | ✅ |
| No external data sources | ✅ |
| Cash Balance ≥ 0 enforced (DQ guard) | ✅ |
| Pure NumPy + Python stdlib | ✅ |
| Signal mathematically explainable | ✅ |
| Single command runs everything | ✅ |
| All output files written to `outputs/` | ✅ |
| Runs within 30-second wall-clock budget | ✅ |

---

## Requirements

```
numpy>=1.24
```

Python ≥ 3.8. No other dependencies.

```bash
pip install -r requirements.txt
```

---

## Recommended Competition Day Workflow

```bash
# ── BEFORE COMPETITION DAY ─────────────────────────────────

# Step 1: Install dependency
pip install -r requirements.txt

# Step 2: Create and validate on synthetic data
python generate_sample_data.py
python run.py --data data/sample_dataset.csv

# Step 3: Profile your hardware
python benchmark.py --data data/sample_dataset.csv --ticker TICKER_00

# Step 4: Find optimal parameters
python parameter_tuner.py --data data/sample_dataset.csv --ticker TICKER_00
# → Check outputs/best_params.txt for best fast/slow/zscore values

# ── ON COMPETITION DAY (15-minute window) ──────────────────

# Option A: Fully automatic (no prior knowledge of column name)
python run.py --data data/competition_data.csv

# Option B: With pre-tuned parameters
python run.py --data data/competition_data.csv --fast 20 --slow 60

# Option C: Column name known + tuned parameters
python run.py --data data/competition_data.csv --ticker TICKER_00 --fast 20 --slow 60
```

---

## License

This project was built for the **MOCCM IITB 2026 Black-Box Signal Hunt** competition.

---

*Built with pure NumPy — no ML libraries, no magic.*
