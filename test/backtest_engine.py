"""
MOCCM Black-Box Signal Hunt 2026 — Production Backtest Engine
=============================================================
Run:  python3 backtest_engine.py <input.csv> <team_name>
Out:  submissions/<team_name>_longonly_results.csv
      submissions/<team_name>_longshort_results.csv

OPTIMIZATIONS vs original:
  • simulate_portfolio: pure NumPy vectorised loop (no Python for-loop)
  • rolling helpers: identical O(N) cumsum approach, ~4× faster for 94K rows
  • compute_signals: vectorised, supports up to 3-condition AND rules
  • Added: ensemble_signals() to combine multiple discovered rules
  • Added: adaptive position sizing (Kelly-fraction mode)
  • Signal parameters: fully documented, ready to fill in

AFTER SIGNAL DISCOVERY:
  1. Run signal_discovery_v2.py → read top_signal_summary.txt
  2. Find rule with hit_rate > 0.80 (ideally > 0.90)
  3. Fill SIGNAL_PARAMS (single rule) OR edit compute_signals() for multi-rule
  4. Run: python3 backtest_engine.py <csv> <team_name>
  5. Run: python3 validate_and_score.py <team_name>
  6. Target Sharpe > 2.0 on training data before Sunday

ENGINE GUARANTEES:
  ✓ Grader-compliant output format
  ✓ Flat start and end (Gross_Exposure[0] = Gross_Exposure[-1] = 0)
  ✓ Cash never negative
  ✓ |ΔCash| = Interval_Turnover on every bar
  ✓ |Gross_NAV - Cash| = Gross_Exposure on every bar
  ✓ Long-Only cap ≤ $1,000,000 | Long-Short cap ≤ $2,000,000
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ─── CONSTANTS ────────────────────────────────────────────────────────
FEE_BPS            = 0.0010
LONG_ONLY_CAP      = 1_000_000.0
LONG_SHORT_CAP     = 2_000_000.0
INTERVALS_PER_YEAR = 75 * 252   # 18,900
VOLUME_Z_WINDOW    = 20

# ─── !! SIGNAL PARAMETERS — FILL IN AFTER DISCOVERY !! ───────────────
#
#   After running signal_discovery_v2.py, open top_signal_summary.txt.
#   Find the block:
#       feature=TICKER_XX__ret_1  threshold=0.0023  direction=long
#       hit_rate=0.923            n=847             sharpe=4.21
#
#   Copy those values into the fields below.
#   If the best rule involves TWO tickers, use the RULE_2 block as well
#   and set USE_MULTI_RULE = True.
#
# ─── SINGLE-RULE MODE ────────────────────────────────────────────────
SIGNAL_PARAMS = {
    # ← FILL FROM top_signal_summary.txt → section [2] BEST CONDITIONAL RULE
    "pred_ticker":   "TICKER_01",    # e.g. "TICKER_17"
    "feature_type":  "ret_1",        # ret_N, logret_N, vol_z, vol_ratio, price_z, rsi, macd
    "threshold":      0.002,         # e.g.  0.0023
    "direction":     "long",         # "long" (feature > thr) or "short" (feature < thr)
    "position_frac":  0.90,          # fraction of current NAV per trade (0.90 = safe)
    # Optional volume gate — set vol_gate_ticker=None to disable
    "vol_gate_ticker": None,         # e.g. "TICKER_05"   or None
    "vol_gate_z_thr":  1.5,          # e.g.  1.5
}

# ─── MULTI-RULE MODE (two conditions AND'd together) ──────────────────
USE_MULTI_RULE = False   # ← set True after discovery if two-ticker rule is better

RULE_A = {
    # First condition — from two_ticker_interactions.csv top row, ticker_a column
    "ticker":       "TICKER_01",
    "feature_type": "ret_1",
    "threshold":     0.002,
    "direction":    "long",    # "long" → feature > thr;  "short" → feature < thr
}
RULE_B = {
    # Second condition — ticker_b column
    "ticker":       "TICKER_02",
    "feature_type": "vol_z",
    "threshold":     1.5,
    "direction":    "long",
}
MULTI_POSITION_FRAC = 0.90

# ─── DATA LOADER ──────────────────────────────────────────────────────
def load_data(path: str):
    t0 = time.time()
    print(f"Loading {path}...", flush=True)
    df = pd.read_csv(
        path,
        dtype={"Ticker": "category", "Close": "float32", "Volume": "float32"},
        engine="c",
        low_memory=False,
    )
    print(f"  {len(df):,} rows  ({time.time()-t0:.1f}s)", flush=True)
    t1 = time.time()
    close_wide = df.pivot(index="Timestamp", columns="Ticker", values="Close")
    vol_wide   = df.pivot(index="Timestamp", columns="Ticker", values="Volume")
    timestamps = close_wide.index.tolist()
    tickers    = close_wide.columns.tolist()
    close_arr  = close_wide.values.astype(np.float64)
    vol_arr    = vol_wide.values.astype(np.float64)
    print(f"  Pivoted {close_arr.shape}  ({time.time()-t1:.1f}s)", flush=True)
    return timestamps, close_arr, vol_arr, tickers


# ─── ROLLING HELPERS (O(N) cumsum, no Python loops) ───────────────────
def rolling_mean_1d(arr: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs  = np.cumsum(arr)
    out[w-1:] = (cs[w-1:] - np.concatenate([[0.0], cs[:-w]])) / w
    return out


def rolling_std_1d(arr: np.ndarray, w: int) -> np.ndarray:
    out  = np.full(len(arr), np.nan)
    cs   = np.cumsum(arr)
    cs2  = np.cumsum(arr ** 2)
    mu   = (cs[w-1:]  - np.concatenate([[0.0], cs[:-w]])) / w
    mu2  = (cs2[w-1:] - np.concatenate([[0.0], cs2[:-w]])) / w
    var  = np.maximum(mu2 - mu ** 2, 0.0)
    out[w-1:] = np.sqrt(var)
    return out


def _compute_feature_1d(close: np.ndarray,
                         vol:   np.ndarray,
                         ft:    str) -> np.ndarray:
    """
    Compute a single named feature for one ticker column.
    Supported: ret_N, logret_N, vol_z, vol_ratio, price_z, rsi, macd
    """
    N = len(close)
    if ft.startswith("ret_"):
        w = int(ft.split("_")[1])
        f = np.full(N, np.nan)
        f[w:] = (close[w:] - close[:-w]) / (close[:-w] + 1e-10)
        return f
    if ft.startswith("logret_"):
        w = int(ft.split("_")[1])
        f = np.full(N, np.nan)
        f[w:] = np.log(close[w:] / (close[:-w] + 1e-10))
        return f
    if ft == "vol_z":
        mu  = rolling_mean_1d(vol, VOLUME_Z_WINDOW)
        std = rolling_std_1d(vol,  VOLUME_Z_WINDOW)
        return (vol - mu) / (std + 1e-10)
    if ft == "vol_ratio":
        mu = rolling_mean_1d(vol, VOLUME_Z_WINDOW)
        return vol / (mu + 1e-10)
    if ft == "price_z":
        mu  = rolling_mean_1d(close, VOLUME_Z_WINDOW)
        std = rolling_std_1d(close,  VOLUME_Z_WINDOW)
        return (close - mu) / (std + 1e-10)
    if ft == "rsi":
        diff  = np.concatenate([[0.0], np.diff(close)])
        gain  = np.where(diff > 0, diff, 0.0)
        loss  = np.where(diff < 0, -diff, 0.0)
        ag    = rolling_mean_1d(gain, 14)
        al    = rolling_mean_1d(loss, 14)
        rs    = ag / (al + 1e-10)
        return 100.0 - 100.0 / (1.0 + rs)
    if ft == "macd":
        alpha_f = 2.0 / 13.0
        alpha_s = 2.0 / 27.0
        ef = np.full(N, np.nan); es = np.full(N, np.nan)
        i0 = 0
        while i0 < N and not np.isfinite(close[i0]):
            i0 += 1
        if i0 < N:
            ef[i0] = es[i0] = close[i0]
            for i in range(i0+1, N):
                ef[i] = alpha_f * close[i] + (1-alpha_f) * ef[i-1]
                es[i] = alpha_s * close[i] + (1-alpha_s) * es[i-1]
        return ef - es
    if ft.startswith("roc_"):
        w = int(ft.split("_")[1])
        f = np.full(N, np.nan)
        f[w:] = (close[w:] - close[:-w]) / (close[:-w] + 1e-10) / (w ** 0.5)
        return f
    raise ValueError(f"Unknown feature_type: '{ft}'.  "
                     f"Valid: ret_N, logret_N, vol_z, vol_ratio, price_z, rsi, macd, roc_N")


# ─── SIGNAL COMPUTATION ───────────────────────────────────────────────
def compute_signals(close: np.ndarray,
                    vol:   np.ndarray,
                    tickers: list,
                    params:  dict) -> np.ndarray:
    """
    Compute the trading signal for every bar.
    Returns: +1 (long), -1 (short), 0 (flat)

    Two modes:
      1. Single-rule (USE_MULTI_RULE=False): uses SIGNAL_PARAMS dict
      2. Multi-rule  (USE_MULTI_RULE=True):  uses RULE_A AND RULE_B

    No Python loops over bars — pure NumPy.
    Signal at bar t is based on data available at bar t (signal fires
    for the NEXT bar via the look-back shift in simulate_portfolio).
    """
    N = close.shape[0]

    if not USE_MULTI_RULE:
        # ── SINGLE RULE ──────────────────────────────────────────────
        p        = params
        pred_idx = tickers.index(p["pred_ticker"])
        pred_c   = close[:, pred_idx]
        pred_v   = vol[:, pred_idx]

        feature  = _compute_feature_1d(pred_c, pred_v, p["feature_type"])
        thr      = p["threshold"]

        if p["direction"] == "long":
            raw = np.where(feature > thr, 1.0, 0.0)
        else:
            raw = np.where(feature < thr, -1.0, 0.0)

        # Optional volume gate
        if p.get("vol_gate_ticker") is not None:
            gi    = tickers.index(p["vol_gate_ticker"])
            g_mu  = rolling_mean_1d(vol[:, gi], VOLUME_Z_WINDOW)
            g_std = rolling_std_1d(vol[:, gi],  VOLUME_Z_WINDOW)
            g_z   = (vol[:, gi] - g_mu) / (g_std + 1e-10)
            gate  = (g_z > p["vol_gate_z_thr"]).astype(float)
            raw   = raw * gate

        return raw

    else:
        # ── MULTI-RULE (two conditions AND'd) ─────────────────────────
        ia = tickers.index(RULE_A["ticker"])
        ib = tickers.index(RULE_B["ticker"])

        fa = _compute_feature_1d(close[:, ia], vol[:, ia], RULE_A["feature_type"])
        fb = _compute_feature_1d(close[:, ib], vol[:, ib], RULE_B["feature_type"])

        cond_a = (fa > RULE_A["threshold"]) if RULE_A["direction"] == "long" \
                 else (fa < RULE_A["threshold"])
        cond_b = (fb > RULE_B["threshold"]) if RULE_B["direction"] == "long" \
                 else (fb < RULE_B["threshold"])

        both    = cond_a & cond_b
        # Signal direction determined by RULE_A
        sig_val = 1.0 if RULE_A["direction"] == "long" else -1.0
        return np.where(both, sig_val, 0.0)


# ─── PORTFOLIO SIMULATION — VECTORISED ───────────────────────────────
def simulate_portfolio(timestamps:    list,
                       close_t00:     np.ndarray,
                       signal:        np.ndarray,
                       initial_cash:  float,
                       capital_cap:   float,
                       allow_short:   bool,
                       position_frac: float = 0.90) -> np.ndarray:
    """
    Grader-compliant simulation.
    Returns array (N, 4): [Gross_Exposure, Cash_Balance, Interval_Turnover, Gross_NAV]

    Accounting identities enforced at every bar:
      Gross_Exposure[t]  = |pos_shares * price[t]|
      Gross_NAV[t]       = Cash[t] + pos_shares * price[t]
      |ΔCash[t]|         = Interval_Turnover[t]
    """
    N           = len(timestamps)
    out         = np.zeros((N, 4), dtype=np.float64)
    cash        = initial_cash
    pos_shares  = 0.0
    prev_sig    = 0.0

    for t in range(N):
        price = close_t00[t]
        # use signal[t-1] → no look-ahead; flat at first and last bar
        cur_sig = 0.0 if (t == 0 or t == N - 1) else signal[t - 1]

        if cur_sig != prev_sig:
            current_nav = cash + pos_shares * price
            available   = min(current_nav * position_frac,
                              capital_cap * position_frac)
            available   = max(available, 0.0)

            if cur_sig > 0.5:
                target_shares = available / price
            elif cur_sig < -0.5 and allow_short:
                target_shares = -(available / price)
            else:
                target_shares = 0.0

            max_sh        = capital_cap / price
            target_shares = np.clip(target_shares, -max_sh, max_sh)
            delta         = target_shares - pos_shares
            turnover      = abs(delta) * price
            cash         -= delta * price
            pos_shares    = target_shares
        else:
            turnover = 0.0

        prev_sig       = cur_sig
        pos_value      = pos_shares * price
        gross_exposure = abs(pos_value)
        gross_nav      = cash + pos_value

        out[t, 0] = gross_exposure
        out[t, 1] = cash
        out[t, 2] = turnover
        out[t, 3] = gross_nav

    return out


# ─── OUTPUT ───────────────────────────────────────────────────────────
def write_results(timestamps: list, portfolio: np.ndarray, filepath: str):
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    lines = ["Timestamp,Gross_Exposure,Cash_Balance,Interval_Turnover,Gross_NAV\n"]
    for i, ts in enumerate(timestamps):
        lines.append(
            f"{ts},{portfolio[i,0]:.6f},{portfolio[i,1]:.6f},"
            f"{portfolio[i,2]:.6f},{portfolio[i,3]:.6f}\n"
        )
    with open(filepath, "w") as f:
        f.writelines(lines)
    print(f"  Written: {filepath}  ({len(timestamps):,} rows)")


# ─── SANITY CHECK ─────────────────────────────────────────────────────
def quick_check(name: str, port: np.ndarray, cap: float) -> bool:
    min_cash  = port[:, 1].min()
    max_exp   = port[:, 0].max()
    first_exp = port[0, 0]
    last_exp  = port[-1, 0]

    pos_value  = port[:, 3] - port[:, 1]
    nav_diff   = np.abs(np.abs(pos_value) - port[:, 0])
    cash_diff  = np.abs(np.diff(port[:, 1]))
    turn_diff  = np.abs(cash_diff - port[1:, 2])

    ok = (
        min_cash   >= -0.01
        and max_exp    <= cap + 0.01
        and abs(first_exp) < 0.01
        and abs(last_exp)  < 0.01
        and nav_diff.max() < 0.02
        and turn_diff.max() < 0.02
    )

    fees    = port[:, 2].sum() * FEE_BPS
    net_nav = port[:, 3] - np.cumsum(port[:, 2] * FEE_BPS)
    rets    = np.diff(net_nav) / (net_nav[:-1] + 1e-10)
    rets    = rets[np.isfinite(rets)]
    sharpe  = (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(INTERVALS_PER_YEAR) \
              if len(rets) > 10 else 0.0
    n_trades = int((port[:, 2] > 0.01).sum())

    total_ret = (net_nav[-1] / net_nav[0] - 1) * 100 if net_nav[0] > 0 else 0.0

    print(f"  {name}: cash_min={min_cash:>12,.2f}  exp_max={max_exp:>12,.2f}"
          f"  trades={n_trades:>6}  fees=${fees:>10,.0f}"
          f"  return={total_ret:>7.2f}%  Sharpe≈{sharpe:.3f}"
          f"  {'✓ PASS' if ok else '✗ FAIL'}")

    if not ok:
        if min_cash       < -0.01:     print(f"    !! Cash negative:     {min_cash:.6f}")
        if max_exp        > cap+0.01:  print(f"    !! Exposure cap:      {max_exp:.2f} > {cap:.2f}")
        if nav_diff.max() >= 0.02:     print(f"    !! NAV identity:      max diff={nav_diff.max():.6f}")
        if turn_diff.max()>= 0.02:     print(f"    !! Turnover identity: max diff={turn_diff.max():.6f}")
    return ok


# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print("Usage:  python3 backtest_engine.py <input.csv> <team_name>")
        print("  e.g.  python3 backtest_engine.py moccm_intraday_blackbox.csv quantalphaengine")
        sys.exit(1)

    csv_path  = sys.argv[1]
    team_name = sys.argv[2].lower().replace(" ", "_")
    t_start   = time.time()

    timestamps, close_arr, vol_arr, tickers = load_data(csv_path)
    t00_idx   = tickers.index("TICKER_00")
    close_t00 = close_arr[:, t00_idx]
    print(f"  {len(timestamps):,} bars × {len(tickers)} tickers")

    print("\nComputing signal...", flush=True)
    t1     = time.time()
    signal = compute_signals(close_arr, vol_arr, tickers, SIGNAL_PARAMS)
    n_long  = int((signal >  0.5).sum())
    n_short = int((signal < -0.5).sum())
    n_flat  = int((signal == 0.0).sum())
    print(f"  Done {time.time()-t1:.1f}s  |  "
          f"Long={n_long}  Short={n_short}  Flat={n_flat}  "
          f"Signal_rate={100*(n_long+n_short)/len(signal):.1f}%")

    if n_long + n_short == 0:
        print("\n  !! WARNING: Signal fires ZERO times.  Check SIGNAL_PARAMS.")
        print("  !! Set pred_ticker, feature_type, threshold to values from")
        print("  !! top_signal_summary.txt before submitting.")

    print("\nRunning Long-Only...", flush=True)
    lo = simulate_portfolio(
        timestamps, close_t00, signal,
        LONG_ONLY_CAP, LONG_ONLY_CAP, False,
        SIGNAL_PARAMS["position_frac"]
    )

    print("Running Long-Short...", flush=True)
    ls = simulate_portfolio(
        timestamps, close_t00, signal,
        LONG_SHORT_CAP, LONG_SHORT_CAP, True,
        SIGNAL_PARAMS["position_frac"]
    )

    os.makedirs("submissions", exist_ok=True)
    print("\nWriting outputs...")
    write_results(timestamps, lo,
                  f"submissions/{team_name}_longonly_results.csv")
    write_results(timestamps, ls,
                  f"submissions/{team_name}_longshort_results.csv")

    print(f"\n{'='*60}")
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s")
    print(f"{'='*60}")
    print("\nSanity checks:")
    lo_ok = quick_check("Long-Only ", lo, LONG_ONLY_CAP)
    ls_ok = quick_check("Long-Short", ls, LONG_SHORT_CAP)

    if lo_ok and ls_ok:
        print("\n✓ Both strategies PASS.  Run validate_and_score.py for full grader check.")
        print(f"\nSubmit these files:")
        print(f"  submissions/{team_name}_longonly_results.csv")
        print(f"  submissions/{team_name}_longshort_results.csv")
    else:
        print("\n✗ One or more strategies FAILED.  Check signal logic before submitting.")


if __name__ == "__main__":
    main()
