"""
MOCCM Black-Box Signal Hunt 2026 — Local Grader
================================================
Mirrors moccm_grader_modified.py exactly.
Run BEFORE submitting to catch any issues.

Usage:
  python3 validate_and_score.py <team_name>
  python3 validate_and_score.py <team_name> --rows <N>

  <N> overrides the expected row count (default 94,500 = 5 × 252 × 75).
  Use this if the judging dataset has a different length.

Examples:
  python3 validate_and_score.py quantalphaengine
  python3 validate_and_score.py quantalphaengine --rows 75600
"""
import pandas as pd
import numpy as np
import sys
import os
import argparse

FEE_BPS            = 0.0010
LONG_ONLY_CAP      = 1_000_000
LONG_SHORT_CAP     = 2_000_000
DEFAULT_ROWS       = 5 * 252 * 75   # 94,500  (competition default)
INTERVALS_PER_YEAR = 75 * 252       # 18,900
ATOL               = 0.01


def validate(df: pd.DataFrame, expected_rows: int) -> tuple:
    required = ["Timestamp", "Gross_Exposure", "Cash_Balance",
                "Interval_Turnover", "Gross_NAV"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing columns: {missing}.  Got: {df.columns.tolist()}"

    n = len(df)
    if n != expected_rows:
        return False, (f"Row count mismatch: expected {expected_rows:,}, got {n:,}.  "
                       f"If judging CSV has different length, rerun with --rows {n}")

    for c in ["Gross_Exposure", "Cash_Balance", "Interval_Turnover", "Gross_NAV"]:
        bad = (~np.isfinite(df[c])).sum()
        if bad > 0:
            return False, f"Non-finite values in '{c}': {bad} rows"

    if (df["Gross_Exposure"] < -ATOL).any():
        idx = (df["Gross_Exposure"] < -ATOL).idxmax()
        return False, f"Gross_Exposure negative at row {idx}: {df['Gross_Exposure'][idx]:.6f}"

    if (df["Interval_Turnover"] < -ATOL).any():
        idx = (df["Interval_Turnover"] < -ATOL).idxmax()
        return False, f"Interval_Turnover negative at row {idx}: {df['Interval_Turnover'][idx]:.6f}"

    ts = pd.to_datetime(df["Timestamp"])
    if not ts.is_monotonic_increasing:
        first_bad = (ts.diff() <= pd.Timedelta(0)).idxmax()
        return False, f"Timestamps not monotonic at row {first_bad}"
    if ts.duplicated().any():
        first_dup = ts.duplicated().idxmax()
        return False, f"Duplicate timestamp at row {first_dup}: {ts[first_dup]}"

    if abs(df["Gross_Exposure"].iloc[0]) > ATOL:
        return False, f"Not flat at start: Gross_Exposure[0] = {df['Gross_Exposure'].iloc[0]:.6f}"
    if abs(df["Gross_Exposure"].iloc[-1]) > ATOL:
        return False, f"Not flat at end: Gross_Exposure[-1] = {df['Gross_Exposure'].iloc[-1]:.6f}"

    # NAV decomposition: |NAV - Cash| = Gross_Exposure
    pos_val = df["Gross_NAV"].values - df["Cash_Balance"].values
    nav_err = np.abs(np.abs(pos_val) - df["Gross_Exposure"].values)
    if (nav_err > ATOL).any():
        idx = int(nav_err.argmax())
        return False, (f"NAV identity broken at row {idx}: "
                       f"|NAV-Cash|={abs(pos_val[idx]):.6f}  "
                       f"Gross_Exposure={df['Gross_Exposure'].iloc[idx]:.6f}  "
                       f"diff={nav_err[idx]:.6f}")

    # Cash equation: |ΔCash[t]| = Interval_Turnover[t]
    cash_diff = np.abs(np.diff(df["Cash_Balance"].values))
    turn_t    = df["Interval_Turnover"].values[1:]
    turn_err  = np.abs(cash_diff - turn_t)
    if (turn_err > ATOL).any():
        idx = int(turn_err.argmax()) + 1
        return False, (f"Cash identity broken at row {idx}: "
                       f"|ΔCash|={cash_diff[idx-1]:.6f}  "
                       f"Interval_Turnover={turn_t[idx-1]:.6f}  "
                       f"diff={turn_err[idx-1]:.6f}")

    return True, "Pass"


def score(path: str, cap: float, expected_rows: int) -> tuple:
    if not os.path.exists(path):
        return False, 0.0, f"File not found: {path}"

    df = pd.read_csv(path)
    ok, msg = validate(df, expected_rows)
    if not ok:
        return False, 0.0, msg

    if (df["Cash_Balance"] < -ATOL).any():
        worst = df["Cash_Balance"].min()
        return False, 0.0, f"Margin Call (Cash went negative: {worst:.6f})"

    max_exp = df["Gross_Exposure"].max()
    if max_exp > cap + ATOL:
        return False, 0.0, f"Capital breach: {max_exp:.2f} > {cap}"

    df["Friction"] = df["Interval_Turnover"] * FEE_BPS
    df["CumFees"]  = df["Friction"].cumsum()
    df["Net_NAV"]  = df["Gross_NAV"] - df["CumFees"]

    if (df["Net_NAV"] <= 0).any():
        return False, 0.0, "Bankrupt (Net_NAV ≤ 0)"

    rets = df["Net_NAV"].pct_change().fillna(0).replace([np.inf, -np.inf], np.nan).dropna()
    if rets.empty or rets.std() == 0:
        return False, 0.0, "No valid returns or zero volatility"

    sharpe      = (rets.mean() / rets.std()) * np.sqrt(INTERVALS_PER_YEAR)
    total_ret   = (df["Net_NAV"].iloc[-1] / df["Net_NAV"].iloc[0] - 1) * 100
    fees        = df["CumFees"].iloc[-1]
    ntrades     = int((df["Interval_Turnover"] > 0.01).sum())
    max_dd      = _max_drawdown(df["Net_NAV"].values)
    final_nav   = df["Net_NAV"].iloc[-1]

    print(f"    Rows checked   : {len(df):,}")
    print(f"    Total return   : {total_ret:.2f}%")
    print(f"    Total fees     : ${fees:,.2f}")
    print(f"    Trade bars     : {ntrades:,}  ({100*ntrades/len(df):.1f}% of bars)")
    print(f"    Max drawdown   : {max_dd:.2f}%")
    print(f"    Final Net NAV  : ${final_nav:,.2f}")
    print(f"    Sharpe ratio   : {sharpe:.4f}")

    return True, round(sharpe, 4), "Pass"


def _max_drawdown(nav: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown in percent."""
    peak = np.maximum.accumulate(nav)
    dd   = (nav - peak) / (peak + 1e-10) * 100.0
    return float(dd.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("team",   nargs="?", default="quantalphaengine")
    parser.add_argument("--rows", type=int,  default=DEFAULT_ROWS,
                        help=f"Expected row count (default {DEFAULT_ROWS})")
    args = parser.parse_args()

    team = args.team
    lo_path = f"submissions/{team}_longonly_results.csv"
    ls_path = f"submissions/{team}_longshort_results.csv"

    sep = "=" * 60
    print(f"\n{sep}\nVALIDATING: {team.upper()}\nExpected rows: {args.rows:,}\n{sep}")

    all_pass = True
    sharpes  = {}

    for label, path, cap in [
        ("Long-Only",  lo_path, LONG_ONLY_CAP),
        ("Long-Short", ls_path, LONG_SHORT_CAP),
    ]:
        print(f"\n{label}: {path}")
        ok, sharpe, msg = score(path, cap, args.rows)
        sharpes[label] = sharpe
        status = "PASS ✓" if ok else "FAIL ✗"
        print(f"  Status : {status}  |  {msg}")
        if not ok:
            all_pass = False

    print(f"\n{sep}")
    if all_pass:
        best_sharpe = max(sharpes.values())
        print(f"RESULT: ALL PASS ✓ — Safe to submit.")
        print(f"Best Sharpe: {best_sharpe:.4f}  "
              f"({'✓ Exceeds target 2.0' if best_sharpe >= 2.0 else '⚠ Below target 2.0'})")
    else:
        print("RESULT: FAILURES DETECTED ✗ — Fix before submitting.")
    print(f"{sep}\n")

    if all_pass:
        print("Files to upload:")
        print(f"  {lo_path}")
        print(f"  {ls_path}")


if __name__ == "__main__":
    main()
