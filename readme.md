# QuantAlphaEngine

Cross-asset lead-lag alpha discovery and fee-adjusted backtesting on intraday price data.

## Overview

Given a 50-column, 189,000-row panel of anonymized intraday prices, this project identifies a single leading series whose lagged returns predict another series' next-bar direction with **57.2% accuracy**, then turns that signal into a fee-adjusted trading strategy with a full accounting simulation behind it — cash, exposure, NAV, and turnover tracked bar-by-bar and checked against invariants (cash never negative, NAV decomposes exactly, first/last bar flat).

The core constraint driving the implementation: the whole pipeline — load, signal, backtest, validate — has to run in single-digit seconds over 189k bars while every bar satisfies the accounting invariants exactly. That's why the EMA is a vectorized IIR filter rather than a loop, and why the two genuinely sequential parts (the signal state machine and the backtest simulation) are Numba-JIT compiled.

## Key Features

- Lead-lag discovery via lagged Spearman IC scan across 50 tickers
- Vectorized EMA (`scipy.signal.lfilter`, single-pass IIR)
- Numba-JIT signal and backtest loops
- Long-Only and Long-Short strategies, independently parameterized
- Fee-adjusted simulation (10 bps per side)
- Capital/margin/exposure caps with intra-run trimming
- Selective CSV ingestion (PyArrow, only 2 of 50 columns read)
- Independent validator script that re-checks every output from scratch

## Repository Structure

```
engine.py                 # load → signal → backtest → validate → report
validate_and_score.py     # independent re-validation of output CSVs
requirements.txt
submissions/
  ├── quantalphaengine_longonly_results.csv
  └── quantalphaengine_longshort_results.csv
test/                     # exploration trail — not part of the production pipeline
  └── top_signal_summary.txt   # lead-lag discovery output (Spearman IC, hit rates)
```

`engine.py` is the entire production pipeline in one file — load, signal, backtest, validate, CLI. `test/` holds earlier discovery scripts and rejected variants (e.g. testing whether a second ticker improves the signal — it doesn't, at usable trade frequency) kept as a record of what was tried and ruled out, not as active code.

## Strategy

Signal is a fast/slow EMA crossover state machine on the *leading* ticker's returns — the traded instrument's own price history isn't used to decide entries or exits.

```
FLAT  → LONG   fast EMA crosses above 0
FLAT  → SHORT  fast EMA crosses below 0     (Long-Short only)
LONG  → FLAT   slow EMA drops below exit threshold
SHORT → FLAT   slow EMA rises above -exit threshold
```

| | Long-Only | Long-Short |
|---|---|---|
| Entry / exit EMA span | 25 / 2,000 | 20 / 1,500 |
| Capital cap | $1,000,000 | $2,000,000 |
| Notional | $850,000 | $1.8M long / $400K short |

Execution is next-bar (no look-ahead); first and last bar of every run are forced flat.

## Performance

189,000 bars, fee-adjusted (10 bps/side):

| | Long-Only | Long-Short |
|---|---|---|
| Sharpe | 0.8155 | 0.4177 |
| Total return | 17.60% | 131.93% |
| Trades | 1,099 | 26 |
| Fees paid | $713,484.79 | $3,322.66 |
| Max drawdown | −99.01% | −66.15% |

Blended Sharpe: 0.6166. Full pipeline runtime: ~9 seconds measured.

## Validation

Every run is checked, both inline and by a second, independent script reading only the output CSVs:

- Cash and turnover never negative
- Flat at open and close
- Exposure never exceeds capital cap
- `|Gross NAV − Cash| = Gross Exposure` (exact, every bar)
- `|ΔCash| =` that bar's turnover (exact, every bar)
- Net NAV stays positive throughout

## Limitations

- **In-sample only** — no walk-forward or holdout split; results describe fit to the data the strategy was designed against.
- Parameters (spans, thresholds, notional) were hand-selected, not cross-validated.
- Single asset pair, not a portfolio. A second ticker was tested as an additional filter and rejected — see `test/`.
- Long-Only's −99% max drawdown means the Sharpe alone understates its risk.
- No slippage or market-impact model — flat 10 bps fee, fills at close.
- Anonymized single-source data; no evidence the signal generalizes elsewhere.
- No live/paper trading, no unit test suite.

## Getting Started

```bash
pip install -r requirements.txt
python engine.py <data.csv> <run_name>
python validate_and_score.py <run_name> --rows <expected_rows>
```

## Technologies

NumPy/pandas for array and I/O work, PyArrow as the CSV parsing backend (needed to read only the required columns out of a ~500MB file), SciPy (`lfilter`) for a vectorized EMA, Numba for the two loops that are inherently sequential and can't be vectorized.
