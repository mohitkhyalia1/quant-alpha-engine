"""
MOCCM Black-Box Signal Hunt 2026 — Discovery v3
================================================
OBJECTIVE: Resolve the organizer's 4-ticker hint.

Known: TICKER_00 (target), TICKER_01 (primary predictor, IC=0.190)
Known: TICKER_09 and TICKER_40 show strong standalone vol-panic signals
       (hit_rate_long 66.3% and 62.3% respectively) but were NEVER tested
       as two-ticker pair partners because they have low raw IC.

This script runs 4 targeted scans:

  SCAN A: T01 × T09 joint conditions on T00 forward returns
  SCAN B: T01 × T40 joint conditions on T00 forward returns
  SCAN C: Rolling T00 correlation with all other tickers → find the 4th
          "correlated" ticker the organizer hints at
  SCAN D: 3-way gate: T01 up AND T09/T40 vol-panic AND 4th ticker
          condition → measure T00 forward return hit_rate and sharpe

Run:
    python3 discovery_v3.py moccm_intraday_blackbox.csv

Outputs:
    d3_t01_t09_interactions.csv
    d3_t01_t40_interactions.csv
    d3_t00_correlations.csv
    d3_threeway_gate.csv
    d3_summary.txt
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.signal import lfilter

warnings.filterwarnings("ignore")

INTERVALS_PER_YEAR = 75 * 252   # 18,900
VOL_Z_WINDOW       = 20
CORR_WINDOWS       = [50, 200, 500, 2000]   # bars for rolling correlation
MIN_OBS            = 30          # minimum signals to report a rule
RET_THRESHOLDS     = [0.001, 0.002, 0.003, 0.005]
VOL_Z_THRESHOLDS   = [1.0, 1.5, 2.0, 2.5]
EMA_SPANS          = [15, 20, 25, 35, 50]   # T01 EMA spans to test


# ── HELPERS ──────────────────────────────────────────────────────────────────

def rolling_mean(a, w):
    cs  = np.cumsum(a)
    out = np.full(len(a), np.nan)
    out[w-1:] = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    return out


def rolling_std(a, w):
    out = np.full(len(a), np.nan)
    cs  = np.cumsum(a)
    cs2 = np.cumsum(a**2)
    mu  = (cs[w-1:] - np.concatenate([[0], cs[:-w]])) / w
    mu2 = (cs2[w-1:] - np.concatenate([[0], cs2[:-w]])) / w
    out[w-1:] = np.sqrt(np.maximum(mu2 - mu**2, 0.0))
    return out


def vol_zscore(v, w=VOL_Z_WINDOW):
    mu  = rolling_mean(v, w)
    std = rolling_std(v, w)
    return (v - mu) / (std + 1e-10)


def ema_vec(x, span):
    alpha = 2.0 / (span + 1)
    return lfilter([alpha], [1.0, -(1.0 - alpha)], np.where(np.isfinite(x), x, 0.0))


def signal_stats(fwd_ret, mask):
    """Given a boolean mask, return (n, mean_ret, hit_rate, sharpe)."""
    rc = fwd_ret[mask]
    n  = len(rc)
    if n < MIN_OBS:
        return None
    mean_r = float(rc.mean())
    std_r  = float(rc.std() + 1e-10)
    hr     = float((rc > 0).mean())
    sh     = mean_r / std_r * np.sqrt(INTERVALS_PER_YEAR)
    return n, mean_r, hr, sh


# ── DATA LOAD ────────────────────────────────────────────────────────────────

def load_data(path):
    print(f"Loading {path} ...", flush=True)
    t0 = time.time()
    df = pd.read_csv(path, usecols=["Timestamp","Ticker","Close","Volume"],
                     dtype={"Ticker":"category","Close":"float32","Volume":"float32"},
                     engine="c")
    print(f"  {len(df):,} rows  ({time.time()-t0:.1f}s)", flush=True)
    close = df.pivot(index="Timestamp", columns="Ticker", values="Close").sort_index()
    vol   = df.pivot(index="Timestamp", columns="Ticker", values="Volume").sort_index()
    print(f"  Shape: {close.shape}", flush=True)
    return close, vol


# ── SCAN A/B: T01 × T09 and T01 × T40 ───────────────────────────────────────

def scan_t01_x_ticker(c00, c01, v01, c_b, v_b, ticker_b_name, fwd):
    """
    For every combo of:
      - T01 EMA span (entry signal proxy)
      - T01 return threshold (momentum direction)
      - ticker_b return threshold AND vol_z threshold (panic / vol filter)
    Compute T00 forward-return stats.
    """
    T = len(c00)
    r01  = np.empty(T); r01[0] = 0.0; r01[1:] = np.diff(c01) / (c01[:-1] + 1e-10)
    rb   = np.empty(T); rb[0]  = 0.0; rb[1:]  = np.diff(c_b)  / (c_b[:-1] + 1e-10)
    vz01 = vol_zscore(v01)
    vzb  = vol_zscore(v_b)

    y_mask = np.isfinite(fwd)
    results = []

    # ── Condition A: T01 EMA > 0 (fast EMA entry signal) ─────────────────
    for span in EMA_SPANS:
        ema01 = ema_vec(r01, span)
        t01_up = ema01 > 0   # T01 in momentum-up regime

        # ── Condition B variants for ticker_b ──────────────────────────────

        # B1: ticker_b large DOWN move + vol spike  →  T00 LONG (contrarian)
        for ret_thr in RET_THRESHOLDS:
            for vz_thr in VOL_Z_THRESHOLDS:
                mask_b  = (rb < -ret_thr) & (vzb > vz_thr)
                mask_3  = y_mask & t01_up & mask_b
                stats   = signal_stats(fwd, mask_3)
                if stats:
                    n, mr, hr, sh = stats
                    results.append({
                        "ticker_b": ticker_b_name,
                        "ema_span": span,
                        "cond_b": f"ret<-{ret_thr}_volz>{vz_thr}",
                        "gate": "T01_ema_up AND Tb_crash_vol",
                        "n": n, "mean_ret": round(mr,7),
                        "hit_rate": round(hr,4), "sharpe": round(sh,4),
                        "abs_sharpe": abs(round(sh,4))
                    })

        # B2: ticker_b large DOWN move only (no vol filter)
        for ret_thr in RET_THRESHOLDS:
            mask_b  = rb < -ret_thr
            mask_3  = y_mask & t01_up & mask_b
            stats   = signal_stats(fwd, mask_3)
            if stats:
                n, mr, hr, sh = stats
                results.append({
                    "ticker_b": ticker_b_name,
                    "ema_span": span,
                    "cond_b": f"ret<-{ret_thr}",
                    "gate": "T01_ema_up AND Tb_down",
                    "n": n, "mean_ret": round(mr,7),
                    "hit_rate": round(hr,4), "sharpe": round(sh,4),
                    "abs_sharpe": abs(round(sh,4))
                })

        # B3: ticker_b vol spike only (any direction) → unusual activity filter
        for vz_thr in VOL_Z_THRESHOLDS:
            mask_b  = vzb > vz_thr
            mask_3  = y_mask & t01_up & mask_b
            stats   = signal_stats(fwd, mask_3)
            if stats:
                n, mr, hr, sh = stats
                results.append({
                    "ticker_b": ticker_b_name,
                    "ema_span": span,
                    "cond_b": f"volz>{vz_thr}",
                    "gate": "T01_ema_up AND Tb_vol_spike",
                    "n": n, "mean_ret": round(mr,7),
                    "hit_rate": round(hr,4), "sharpe": round(sh,4),
                    "abs_sharpe": abs(round(sh,4))
                })

        # B4: ticker_b large UP move + vol spike (momentum alignment)
        for ret_thr in RET_THRESHOLDS:
            for vz_thr in VOL_Z_THRESHOLDS:
                mask_b  = (rb > ret_thr) & (vzb > vz_thr)
                mask_3  = y_mask & t01_up & mask_b
                stats   = signal_stats(fwd, mask_3)
                if stats:
                    n, mr, hr, sh = stats
                    results.append({
                        "ticker_b": ticker_b_name,
                        "ema_span": span,
                        "cond_b": f"ret>{ret_thr}_volz>{vz_thr}",
                        "gate": "T01_ema_up AND Tb_up_vol",
                        "n": n, "mean_ret": round(mr,7),
                        "hit_rate": round(hr,4), "sharpe": round(sh,4),
                        "abs_sharpe": abs(round(sh,4))
                    })

    # ── Baseline: T01 EMA up only (no ticker_b filter) ───────────────────
    for span in EMA_SPANS:
        ema01  = ema_vec(r01, span)
        t01_up = ema01 > 0
        mask_3 = y_mask & t01_up
        stats  = signal_stats(fwd, mask_3)
        if stats:
            n, mr, hr, sh = stats
            results.append({
                "ticker_b": "BASELINE",
                "ema_span": span,
                "cond_b": "none",
                "gate": "T01_ema_up_only",
                "n": n, "mean_ret": round(mr,7),
                "hit_rate": round(hr,4), "sharpe": round(sh,4),
                "abs_sharpe": abs(round(sh,4))
            })

    return pd.DataFrame(results)


# ── SCAN C: Rolling T00 correlations with all tickers ────────────────────────

def scan_t00_correlations(close_arr, tickers, t00_idx):
    """
    For each ticker j, compute:
      1. Full-sample Pearson correlation of returns with T00 returns
      2. Rolling correlation (multiple windows)
      3. Conditional correlation: during T01 up regime vs T01 down regime
    Returns ranked DataFrame.
    """
    T    = close_arr.shape[0]
    c00  = close_arr[:, t00_idx].astype(np.float64)
    r00  = np.empty(T); r00[0] = 0.0
    r00[1:] = np.diff(c00) / (c00[:-1] + 1e-10)

    # T01 regime flag
    c01_idx = tickers.index("TICKER_01")
    c01  = close_arr[:, c01_idx].astype(np.float64)
    r01  = np.empty(T); r01[0] = 0.0
    r01[1:] = np.diff(c01) / (c01[:-1] + 1e-10)
    ema_t01 = ema_vec(r01, 25)
    t01_up_flag = ema_t01 > 0

    results = []
    for j, tk in enumerate(tickers):
        if j == t00_idx:
            continue
        c_j = close_arr[:, j].astype(np.float64)
        r_j = np.empty(T); r_j[0] = 0.0
        r_j[1:] = np.diff(c_j) / (c_j[:-1] + 1e-10)

        # Full-sample correlation
        mask  = np.isfinite(r00) & np.isfinite(r_j)
        if mask.sum() < 100:
            continue
        full_corr = float(np.corrcoef(r00[mask], r_j[mask])[0,1])

        # T01-up conditional correlation
        mask_up = mask & t01_up_flag
        mask_dn = mask & ~t01_up_flag
        corr_up = float(np.corrcoef(r00[mask_up], r_j[mask_up])[0,1]) if mask_up.sum() > 100 else np.nan
        corr_dn = float(np.corrcoef(r00[mask_dn], r_j[mask_dn])[0,1]) if mask_dn.sum() > 100 else np.nan

        # Correlation change between regimes (signal of regime-dependence)
        corr_delta = corr_up - corr_dn if (np.isfinite(corr_up) and np.isfinite(corr_dn)) else np.nan

        results.append({
            "ticker": tk,
            "full_corr": round(full_corr, 4),
            "corr_t01_up": round(corr_up, 4) if np.isfinite(corr_up) else None,
            "corr_t01_dn": round(corr_dn, 4) if np.isfinite(corr_dn) else None,
            "corr_delta_up_minus_dn": round(corr_delta, 4) if np.isfinite(corr_delta) else None,
            "abs_full_corr": abs(round(full_corr, 4)),
        })

    df = pd.DataFrame(results).sort_values("abs_full_corr", ascending=False)
    return df


# ── SCAN D: 3-way gate ────────────────────────────────────────────────────────

def scan_threeway(c00, c01, v01, c09, v09, c40, v40, fwd, top_corr_tickers,
                  close_arr, tickers):
    """
    Gate: T01 EMA > 0 (span 25)
    AND EITHER T09 crash+vol OR T40 crash+vol
    AND 4th ticker condition (from top_corr_tickers)

    Also tests T09 AND T40 simultaneously.
    """
    T = len(c00)
    r01  = np.empty(T); r01[0] = 0.0; r01[1:] = np.diff(c01) / (c01[:-1]+1e-10)
    r09  = np.empty(T); r09[0] = 0.0; r09[1:] = np.diff(c09) / (c09[:-1]+1e-10)
    r40  = np.empty(T); r40[0] = 0.0; r40[1:] = np.diff(c40) / (c40[:-1]+1e-10)
    vz09 = vol_zscore(v09)
    vz40 = vol_zscore(v40)

    ema25 = ema_vec(r01, 25)
    t01_up = ema25 > 0
    y_mask = np.isfinite(fwd)

    results = []

    # Panic conditions on T09 and T40
    t09_panic  = (r09 < -0.003) & (vz09 > 1.5)
    t40_panic  = (r40 < -0.003) & (vz40 > 1.5)
    t09_or_t40 = t09_panic | t40_panic
    t09_and_t40 = t09_panic & t40_panic

    for panic_label, panic_mask in [
        ("T09_panic", t09_panic),
        ("T40_panic", t40_panic),
        ("T09_OR_T40", t09_or_t40),
        ("T09_AND_T40", t09_and_t40),
    ]:
        # 2-way: T01 up + panic
        m2 = y_mask & t01_up & panic_mask
        s  = signal_stats(fwd, m2)
        if s:
            n, mr, hr, sh = s
            results.append({
                "gate": f"T01_up + {panic_label}",
                "ticker_4th": "none",
                "cond_4th": "none",
                "n": n, "mean_ret": round(mr,7),
                "hit_rate": round(hr,4), "sharpe": round(sh,4),
                "abs_sharpe": abs(round(sh,4))
            })

        # 3-way: add 4th ticker conditions
        for tk4_name in top_corr_tickers[:10]:
            if tk4_name not in tickers:
                continue
            j4   = tickers.index(tk4_name)
            c4   = close_arr[:, j4].astype(np.float64)
            r4   = np.empty(T); r4[0] = 0.0
            r4[1:] = np.diff(c4) / (c4[:-1]+1e-10)
            vz4  = vol_zscore(close_arr[:, j4].astype(np.float64))  # reuse arr

            for ret_thr in [0.001, 0.002, 0.005]:
                for direction, cond4, label4 in [
                    (1,  r4 > ret_thr,      f"r4>{ret_thr}"),
                    (-1, r4 < -ret_thr,     f"r4<-{ret_thr}"),
                ]:
                    m3 = y_mask & t01_up & panic_mask & cond4
                    s3 = signal_stats(fwd, m3)
                    if s3:
                        n, mr, hr, sh = s3
                        results.append({
                            "gate": f"T01_up + {panic_label}",
                            "ticker_4th": tk4_name,
                            "cond_4th": label4,
                            "n": n, "mean_ret": round(mr,7),
                            "hit_rate": round(hr,4), "sharpe": round(sh,4),
                            "abs_sharpe": abs(round(sh,4))
                        })

    return pd.DataFrame(results)


# ── SUMMARY ──────────────────────────────────────────────────────────────────

def write_summary(df_ab09, df_ab40, df_corr, df_3way, baseline_hr):
    lines = ["="*70,
             "DISCOVERY V3 — 4-TICKER SIGNAL HUNT SUMMARY",
             "="*70, ""]

    lines.append("[A] T01 × T09  —  Best gates (sorted by hit_rate, T01_ema_up AND T09_*):")
    d = df_ab09[df_ab09["ticker_b"] != "BASELINE"].sort_values("hit_rate", ascending=False).head(15)
    lines.append(d[["ema_span","cond_b","n","mean_ret","hit_rate","sharpe"]].to_string(index=False))

    lines.append("")
    lines.append("[B] T01 × T40  —  Best gates (sorted by hit_rate):")
    d = df_ab40[df_ab40["ticker_b"] != "BASELINE"].sort_values("hit_rate", ascending=False).head(15)
    lines.append(d[["ema_span","cond_b","n","mean_ret","hit_rate","sharpe"]].to_string(index=False))

    lines.append("")
    lines.append("[C] T00 Correlation Ranking (all tickers, full-sample + regime-conditional):")
    lines.append(df_corr.head(20).to_string(index=False))

    lines.append("")
    lines.append("[D] 3-way gate results (sorted by hit_rate):")
    d = df_3way.sort_values("hit_rate", ascending=False).head(25)
    lines.append(d.to_string(index=False))

    lines.append("")
    lines.append(f"BASELINE (T01 EMA span=25 up only): hit_rate ~ {baseline_hr:.4f}")

    lines.append("")
    lines.append("INTERPRETATION:")
    lines.append("  hit_rate > baseline → gate ADDS value (use as signal filter)")
    lines.append("  hit_rate >> 0.60   → strong gate, integrate into engine")
    lines.append("  If 3-way gate hit_rate > 0.65 with n > 50 → implement in engine")
    lines.append("="*70)

    summary = "\n".join(lines)
    print("\n" + summary)
    with open("d3_summary.txt", "w", encoding="utf-8") as f:
        f.write(summary)


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 discovery_v3.py moccm_intraday_blackbox.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    t_total  = time.time()

    close_df, vol_df = load_data(csv_path)
    tickers   = close_df.columns.tolist()
    close_arr = close_df.values.astype(np.float64)
    vol_arr   = vol_df.values.astype(np.float64)
    del close_df, vol_df

    t00_idx = tickers.index("TICKER_00")
    t01_idx = tickers.index("TICKER_01")
    t09_idx = tickers.index("TICKER_09")
    t40_idx = tickers.index("TICKER_40")

    c00 = close_arr[:, t00_idx]
    c01 = close_arr[:, t01_idx]
    c09 = close_arr[:, t09_idx]
    c40 = close_arr[:, t40_idx]
    v01 = vol_arr[:, t01_idx]
    v09 = vol_arr[:, t09_idx]
    v40 = vol_arr[:, t40_idx]

    T   = len(c00)
    fwd = np.full(T, np.nan)
    fwd[:-1] = (c00[1:] - c00[:-1]) / (c00[:-1] + 1e-10)

    # ── SCAN A: T01 × T09 ────────────────────────────────────────────────
    print("\n=== SCAN A: T01 × T09 ===", flush=True)
    df_ab09 = scan_t01_x_ticker(c00, c01, v01, c09, v09, "TICKER_09", fwd)
    df_ab09.sort_values("hit_rate", ascending=False, inplace=True)
    df_ab09.to_csv("d3_t01_t09_interactions.csv", index=False)
    print(f"  {len(df_ab09)} conditions tested. Best hit_rate: {df_ab09['hit_rate'].max():.4f}")

    # ── SCAN B: T01 × T40 ────────────────────────────────────────────────
    print("\n=== SCAN B: T01 × T40 ===", flush=True)
    df_ab40 = scan_t01_x_ticker(c00, c01, v01, c40, v40, "TICKER_40", fwd)
    df_ab40.sort_values("hit_rate", ascending=False, inplace=True)
    df_ab40.to_csv("d3_t01_t40_interactions.csv", index=False)
    print(f"  {len(df_ab40)} conditions tested. Best hit_rate: {df_ab40['hit_rate'].max():.4f}")

    # ── SCAN C: T00 correlations ──────────────────────────────────────────
    print("\n=== SCAN C: T00 rolling correlations ===", flush=True)
    df_corr = scan_t00_correlations(close_arr, tickers, t00_idx)
    df_corr.to_csv("d3_t00_correlations.csv", index=False)
    print(df_corr.head(10).to_string(index=False))

    # Top candidates for 4th ticker (exclude T00, T01, T09, T40)
    skip = {"TICKER_00", "TICKER_01", "TICKER_09", "TICKER_40"}
    top_corr_tickers = [r["ticker"] for _, r in df_corr.iterrows()
                        if r["ticker"] not in skip][:15]
    print(f"\n  4th-ticker candidates: {top_corr_tickers[:8]}")

    # ── SCAN D: 3-way gate ───────────────────────────────────────────────
    print("\n=== SCAN D: 3-way gate T01+T09/T40+4th ticker ===", flush=True)
    df_3way = scan_threeway(c00, c01, v01, c09, v09, c40, v40, fwd,
                            top_corr_tickers, close_arr, tickers)
    df_3way.sort_values("hit_rate", ascending=False, inplace=True)
    df_3way.to_csv("d3_threeway_gate.csv", index=False)
    print(f"  {len(df_3way)} 3-way conditions tested.")

    # Baseline hit_rate (T01 span=25 up only)
    r01   = np.empty(T); r01[0] = 0.0; r01[1:] = np.diff(c01)/(c01[:-1]+1e-10)
    ema25 = ema_vec(r01, 25)
    baseline_mask = np.isfinite(fwd) & (ema25 > 0)
    baseline_hr   = float((fwd[baseline_mask] > 0).mean())

    # ── Summary ──────────────────────────────────────────────────────────
    write_summary(df_ab09, df_ab40, df_corr, df_3way, baseline_hr)

    print(f"\nTotal runtime: {time.time()-t_total:.1f}s")
    print("Outputs: d3_t01_t09_interactions.csv  d3_t01_t40_interactions.csv")
    print("         d3_t00_correlations.csv  d3_threeway_gate.csv  d3_summary.txt")


if __name__ == "__main__":
    main()
