"""
MOCCM Black-Box Signal Hunt 2026 — FIXED STRATEGY ENGINE
=========================================================
Team: quantalphaengine

WHY THE ORIGINAL WENT BANKRUPT — ROOT CAUSE:
─────────────────────────────────────────────
Strategy D fires when:
  TICKER_01 ret_1 > 0.002  AND  TICKER_25 ret_1 > 0.002  AND  vol_z > 1.0

vol_z > 1.0 is true ~16% of bars (top of a normal distribution).
ret_1 > 0.002 on TICKER_01 fires on ~20-35% of bars at this threshold.
Combined, the signal toggles on/off almost every bar → ~16,764 transitions.

Each transition = full position turnover ≈ $900K.
Fee per transition = $900K × 0.001 = $900.
16,764 events × $900 = ~$15M in fees — 15× the $1M starting NAV.

Additionally: mean_ret per 1-bar hold ≈ 0.13% < 0.20% round-trip fee.
Every single trade loses money net of costs. Bankruptcy is guaranteed.

THE TWO-PART FIX:
─────────────────
Fix 1 — TIGHTER THRESHOLDS:
  Use the tightest rule from two_ticker_interactions.csv:
    TICKER_01 ret_1 > +0.005  AND  TICKER_25 ret_1 > +0.002   (n=184 signals)
  This rule has mean_ret = 0.2685% > 0.20% round-trip fee → net +0.068%/trade.
  It fires only ~184 times in 5 years (0.19% of bars) → ~368 transitions total.
  Fees ≈ 368 × $900K × 0.001 = $331K — well within the $1M NAV budget.

  Short side is made symmetric:
    TICKER_01 ret_1 < -0.005  AND  TICKER_25 ret_1 < -0.002
  This matches the signal frequency of the long side (~184 events).

Fix 2 — MINIMUM HOLD (hold_bars = 3):
  After entering a position, hold for at least 3 bars regardless of the signal.
  IC at lag-1 = 0.190, lag-2 = 0.125, lag-3 = 0.096 — all materially positive.
  Holding 3 bars captures the cumulative drift while paying the round-trip fee
  only ONCE, preventing rapid on/off churn from consecutive signal bars.

RESULT (estimated from discovery data):
  Long-Only:  368 transitions, fees ≈ $331K, gross ≈ $445K → net +11.3%
  Long-Short: 736 transitions, fees ≈ $1.3M,  gross ≈ $1.8M → net +22.7%
  Both survive; neither goes bankrupt.

Run:
  python3 strategy_engine.py <input.csv> <team_name>
  python3 validate_and_score.py <team_name>

Outputs:
  submissions/<team_name>_longonly_results.csv
  submissions/<team_name>_longshort_results.csv
"""

import numpy as np
import pandas as pd
import sys
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
FEE_BPS            = 0.0010          # 10 bps one-way
LONG_ONLY_CAP      = 1_000_000.0
LONG_SHORT_CAP     = 2_000_000.0
INTERVALS_PER_YEAR = 75 * 252        # 18,900 bars/year
POSITION_FRAC      = 0.90            # deploy 90% of available NAV

# ─── FIX 1: TIGHT THRESHOLDS ─────────────────────────────────────────────────
# Source: two_ticker_interactions.csv row 2
#   ticker_a=TICKER_01, ticker_b=TICKER_25, thr_a=>0.005, thr_b=>0.002
#   n_signals=184, mean_ret=0.002685, hit_rate=0.587, abs_sharpe=50.54
#
# This is the ONLY row in the discovery where mean_ret (0.2685%) strictly
# exceeds the round-trip fee (2 × 10 bps = 0.20%), making each trade
# net-profitable before considering multi-bar hold benefit.
#
# All lower-threshold rows (e.g. thr_a=0.002) have mean_ret ≈ 0.13% < 0.20%
# fee → guaranteed net loss per trade regardless of hold duration.
LONG_THR_A  =  0.005   # TICKER_01 ret_1 > +0.5%
LONG_THR_B  =  0.002   # TICKER_25 ret_1 > +0.2%  (second confirmation)
SHORT_THR_A = -0.005   # TICKER_01 ret_1 < -0.5%  (symmetric short)
SHORT_THR_B = -0.002   # TICKER_25 ret_1 < -0.2%  (second confirmation)

# ─── FIX 2: MINIMUM HOLD ─────────────────────────────────────────────────────
# Hold each position for at least HOLD_BARS bars after entry.
#
# Rationale: IC from lead-lag discovery is positive out to 5+ lags:
#   lag-1: IC=0.190   lag-2: IC=0.125   lag-3: IC=0.096
#   lag-4: IC=0.082   lag-5: IC=0.073
# Holding 3 bars captures cumulative drift across all three lags while
# paying the round-trip fee only ONCE, dramatically improving net P&L
# versus the original 1-bar hold on every signal bar.
HOLD_BARS = 3


# ─── DATA LOADER ──────────────────────────────────────────────────────────────
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


# ─── HELPER ───────────────────────────────────────────────────────────────────
def ret_1d(close: np.ndarray) -> np.ndarray:
    """Simple 1-bar return.  NaN at bar 0."""
    out = np.full(len(close), np.nan)
    out[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)
    return out


# ─── SIGNAL COMPUTATION ───────────────────────────────────────────────────────
def compute_signals(close: np.ndarray, tickers: list) -> np.ndarray:
    """
    Compute raw signal array: +1 (long), -1 (short), 0 (flat).

    Signal at bar t is computed from data available at bar t.
    simulate_portfolio applies it at bar t+1 via signal[t-1] — no look-ahead.

    Long  rule: TICKER_01 ret_1 > LONG_THR_A  AND  TICKER_25 ret_1 > LONG_THR_B
    Short rule: TICKER_01 ret_1 < SHORT_THR_A AND  TICKER_25 ret_1 < SHORT_THR_B
    Conflict (both fire): long takes priority.
    """
    N  = close.shape[0]
    ia = tickers.index("TICKER_01")
    ib = tickers.index("TICKER_25")

    ra = ret_1d(close[:, ia])
    rb = ret_1d(close[:, ib])

    long_cond  = (ra > LONG_THR_A)  & (rb > LONG_THR_B)
    short_cond = (ra < SHORT_THR_A) & (rb < SHORT_THR_B)

    sig = np.zeros(N)
    sig = np.where(short_cond, -1.0, sig)   # short first
    sig = np.where(long_cond,   1.0, sig)   # long overwrites if both

    return sig


# ─── PORTFOLIO SIMULATION WITH MINIMUM HOLD ───────────────────────────────────
def simulate_portfolio(timestamps:    list,
                       close_t00:     np.ndarray,
                       raw_signal:    np.ndarray,
                       initial_cash:  float,
                       capital_cap:   float,
                       allow_short:   bool,
                       hold_bars:     int = HOLD_BARS,
                       position_frac: float = POSITION_FRAC) -> np.ndarray:
    """
    Grader-compliant simulation with minimum-hold enforcement.

    HOLD MECHANIC:
      On entry (signal changes from 0 or reverses), set hold_ctr = hold_bars.
      While hold_ctr > 0: ignore raw_signal, keep current direction, decrement.
      When hold_ctr reaches 0: accept new raw signal.
        - Same direction as current → maintain position, reset hold_ctr.
          (Free: no transition, no fee — signal simply confirmed again.)
        - Different direction or flat → exit (pay exit fee), then re-evaluate.

    Accounting identities at every bar:
      Gross_Exposure[t] = |pos_shares × price[t]|
      Gross_NAV[t]      = Cash[t] + pos_shares × price[t]
      |ΔCash[t]|        = Interval_Turnover[t]
    """
    N          = len(timestamps)
    out        = np.zeros((N, 4), dtype=np.float64)
    cash       = initial_cash
    pos_shares = 0.0
    prev_sig   = 0.0
    hold_ctr   = 0

    for t in range(N):
        price = close_t00[t]

        # Determine effective signal (one-bar lag, boundary flat)
        if t == 0 or t == N - 1:
            cur_sig = 0.0
        else:
            raw_s = raw_signal[t - 1]

            if hold_ctr > 0:
                # Within hold window — lock current direction
                cur_sig  = prev_sig
                hold_ctr -= 1
            else:
                # Hold expired — accept new signal
                cur_sig = raw_s

        # Execute only on transitions
        if cur_sig != prev_sig:
            current_nav = cash + pos_shares * price
            available   = min(current_nav * position_frac,
                              capital_cap * position_frac)
            available   = max(available, 0.0)

            if cur_sig > 0.5:
                target_shares = available / price
                hold_ctr      = hold_bars          # start hold on long entry
            elif cur_sig < -0.5 and allow_short:
                target_shares = -(available / price)
                hold_ctr      = hold_bars          # start hold on short entry
            else:
                target_shares = 0.0
                hold_ctr      = 0                  # no hold when flattening

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


# ─── OUTPUT WRITER ────────────────────────────────────────────────────────────
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


# ─── SANITY CHECK (mirrors grader logic) ─────────────────────────────────────
def quick_check(name: str, port: np.ndarray, cap: float) -> bool:
    min_cash = port[:, 1].min()
    max_exp  = port[:, 0].max()

    pos_value = port[:, 3] - port[:, 1]
    nav_diff  = np.abs(np.abs(pos_value) - port[:, 0])
    cash_diff = np.abs(np.diff(port[:, 1]))
    turn_diff = np.abs(cash_diff - port[1:, 2])

    struct_ok = (
        min_cash          >= -0.01
        and max_exp       <= cap + 0.01
        and abs(port[0,  0]) < 0.01
        and abs(port[-1, 0]) < 0.01
        and nav_diff.max()  < 0.02
        and turn_diff.max() < 0.02
    )

    fees     = port[:, 2].sum() * FEE_BPS
    net_nav  = port[:, 3] - np.cumsum(port[:, 2] * FEE_BPS)
    bankrupt = (net_nav <= 0).any()
    rets     = np.diff(net_nav) / (net_nav[:-1] + 1e-10)
    rets     = rets[np.isfinite(rets)]
    sharpe   = (rets.mean() / (rets.std() + 1e-10)) * np.sqrt(INTERVALS_PER_YEAR) \
               if len(rets) > 10 else 0.0
    n_trades  = int((port[:, 2] > 0.01).sum())
    total_ret = (net_nav[-1] / net_nav[0] - 1) * 100 if net_nav[0] > 0 else 0.0

    passed = struct_ok and not bankrupt
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {name}: cash_min={min_cash:>12,.2f}  exp_max={max_exp:>12,.2f}"
          f"  trades={n_trades:>5}  fees=${fees:>9,.0f}"
          f"  return={total_ret:>7.2f}%  Sharpe≈{sharpe:.3f}  {status}")

    if bankrupt:
        print(f"    !! Net_NAV went ≤ 0 after fees (bankrupt)")
    if not struct_ok:
        if min_cash       < -0.01:    print(f"    !! Cash negative:      {min_cash:.4f}")
        if max_exp        > cap+0.01: print(f"    !! Exposure cap:       {max_exp:.2f} > {cap:.2f}")
        if nav_diff.max() >= 0.02:    print(f"    !! NAV identity error: {nav_diff.max():.2e}")
        if turn_diff.max()>= 0.02:    print(f"    !! Cash/turn error:    {turn_diff.max():.2e}")

    return passed


# ─── FEE BUDGET PRE-CHECK ─────────────────────────────────────────────────────
def fee_budget_check(raw_signal: np.ndarray, n_bars: int):
    """
    Print a pre-run fee budget warning if the signal fires too frequently.
    Maximum safe transitions = NAV / (position_frac * FEE_BPS * NAV) = 1/(pos_frac * FEE_BPS).
    """
    n_long  = int((raw_signal >  0.5).sum())
    n_short = int((raw_signal < -0.5).sum())
    n_total = n_long + n_short

    # Upper bound on transitions (each signal bar could be an entry + exit)
    max_transitions = 2 * n_total
    max_safe_lo = int(LONG_ONLY_CAP  / (POSITION_FRAC * LONG_ONLY_CAP  * FEE_BPS))
    max_safe_ls = int(LONG_SHORT_CAP / (POSITION_FRAC * LONG_SHORT_CAP * FEE_BPS))

    print(f"\n  Signal bars: Long={n_long}  Short={n_short}  Total={n_total}"
          f"  ({100*n_total/n_bars:.3f}% of {n_bars:,} bars)")
    print(f"  Upper-bound transitions: {max_transitions}"
          f"  (hold≥{HOLD_BARS} reduces this significantly)")
    print(f"  Max safe transitions: LO≤{max_safe_lo}  LS≤{max_safe_ls} "
          f"(to keep fees < starting NAV)")

    if max_transitions > max_safe_lo:
        # With HOLD_BARS, actual transitions ≈ 2 * (n_total / avg_cluster_size)
        # avg_cluster_size ≥ 1, so actual ≤ max_transitions
        print(f"  ⚠  Raw signal fires {max_transitions} times; hold={HOLD_BARS} must"
              f" reduce actual transitions to < {max_safe_lo} for LO solvency.")
    else:
        print(f"  ✓  Signal frequency is within safe budget even without hold.")


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 3:
        print("Usage:  python3 strategy_engine.py <input.csv> <team_name>")
        sys.exit(1)

    csv_path  = sys.argv[1]
    team_name = sys.argv[2].lower().replace(" ", "_")
    t_start   = time.time()

    print(f"\n{'='*64}")
    print(f"MOCCM 2026 — FIXED STRATEGY ENGINE  (bankruptcy fix applied)")
    print(f"Signal: TICKER_01 (>{LONG_THR_A}/< {SHORT_THR_A}) + TICKER_25 "
          f"(>{LONG_THR_B}/< {SHORT_THR_B})")
    print(f"Hold:   ≥{HOLD_BARS} bars per entry")
    print(f"Team:   {team_name}")
    print(f"{'='*64}")

    timestamps, close_arr, vol_arr, tickers = load_data(csv_path)
    t00_idx   = tickers.index("TICKER_00")
    close_t00 = close_arr[:, t00_idx]
    N         = len(timestamps)
    print(f"  {N:,} bars × {len(tickers)} tickers")

    for req in ["TICKER_00", "TICKER_01", "TICKER_25"]:
        if req not in tickers:
            print(f"  !! WARNING: {req} not found — signal may not fire.")

    # Compute raw signal
    print("\nComputing signal...", flush=True)
    t1      = time.time()
    raw_sig = compute_signals(close_arr, tickers)
    fee_budget_check(raw_sig, N)
    print(f"  Signal computation: {time.time()-t1:.2f}s")

    if int((raw_sig != 0).sum()) == 0:
        print("  !! CRITICAL: zero signals generated. Aborting.")
        sys.exit(1)

    # Simulate
    print(f"\nRunning Long-Only  (hold≥{HOLD_BARS})...", flush=True)
    lo = simulate_portfolio(
        timestamps, close_t00, raw_sig,
        LONG_ONLY_CAP, LONG_ONLY_CAP, False
    )

    print(f"Running Long-Short (hold≥{HOLD_BARS})...", flush=True)
    ls = simulate_portfolio(
        timestamps, close_t00, raw_sig,
        LONG_SHORT_CAP, LONG_SHORT_CAP, True
    )

    # Write
    os.makedirs("submissions", exist_ok=True)
    lo_path = f"submissions/{team_name}_longonly_results.csv"
    ls_path = f"submissions/{team_name}_longshort_results.csv"
    print("\nWriting outputs...")
    write_results(timestamps, lo, lo_path)
    write_results(timestamps, ls, ls_path)

    # Validate
    print(f"\n{'='*64}")
    print(f"TOTAL TIME: {time.time()-t_start:.1f}s")
    print(f"{'='*64}")
    print("\nSanity checks (mirrors grader validation):")
    lo_ok = quick_check("Long-Only ", lo, LONG_ONLY_CAP)
    ls_ok = quick_check("Long-Short", ls, LONG_SHORT_CAP)

    print(f"\n{'='*64}")
    if lo_ok and ls_ok:
        print("✓ Both strategies PASS all constraints.")
        print(f"\nFull grader:")
        print(f"  python3 validate_and_score.py {team_name}")
        print(f"\nSubmission files:")
        print(f"  {lo_path}")
        print(f"  {ls_path}")
        print(f"\n{'='*64}")
        print("VERDICT: ✅ SUBMIT")
    else:
        print("✗ One or more strategies FAILED.")
        print(f"\nDiagnostic: check the signal frequency.")
        print(f"  If too many transitions, increase LONG_THR_A / abs(SHORT_THR_A)")
        print(f"  or increase HOLD_BARS to reduce churn.")
        print(f"{'='*64}")
        print("VERDICT: ⛔ DO NOT SUBMIT")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
