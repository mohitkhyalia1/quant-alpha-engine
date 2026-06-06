import pandas as pd
import numpy as np
import os
import glob

# --- CONSTRAINTS & SECURITY (same as standard grader) ---
FEE_BPS = 0.0010
LONG_ONLY_CAP = 1_000_000
LONG_SHORT_CAP = 2_000_000
EXPECTED_ROWS = 5 * 252 * 75
INTERVALS_PER_YEAR = 75 * 252

# --- Tolerance for invariant checks ---
ATOL_DOLLAR = 0.01   # 1¢ — direct identities (NAV decomp, cash equation, etc.)


def validate_structure(df):
    """Run structural invariant checks. Returns (ok, message)."""
    required = ["Timestamp", "Gross_Exposure", "Cash_Balance", "Interval_Turnover", "Gross_NAV"]
    if not all(c in df.columns for c in required):
        return False, "Missing required columns."

    n = len(df)
    if n != EXPECTED_ROWS:
        return False, f"Invalid row count. Expected {EXPECTED_ROWS}, got {n}."

    # NaN / Inf rejection
    for c in ["Gross_Exposure", "Cash_Balance", "Interval_Turnover", "Gross_NAV"]:
        if not np.isfinite(df[c]).all():
            return False, f"Non-finite values in column '{c}'."

    # Non-negativity
    if (df["Gross_Exposure"] < -ATOL_DOLLAR).any():
        return False, "Gross_Exposure has negative values."
    if (df["Interval_Turnover"] < -ATOL_DOLLAR).any():
        return False, "Interval_Turnover has negative values."

    # Timestamp validation (no comparison against external dataset)
    try:
        ts = pd.to_datetime(df["Timestamp"])
    except Exception:
        return False, "Timestamp column not parseable as datetime."
    if not ts.is_monotonic_increasing:
        return False, "Timestamps not strictly increasing."
    if ts.duplicated().any():
        return False, "Timestamps contain duplicates."

    # First-bar sanity: must start flat → forces Cash_Balance[0] == Gross_NAV[0] via NAV decomp.
    if abs(df["Gross_Exposure"].iloc[0]) > ATOL_DOLLAR:
        return False, "First bar must be flat (Gross_Exposure[0] != 0)."

    # Last bar must be flat (no open position at end of judging window)
    if abs(df["Gross_Exposure"].iloc[-1]) > ATOL_DOLLAR:
        return False, "Last bar must be flat (Gross_Exposure[-1] != 0)."

    # Invariant 1 — NAV decomposition: |Gross_NAV - Cash_Balance| = Gross_Exposure
    pos_value = df["Gross_NAV"].values - df["Cash_Balance"].values
    diff = np.abs(np.abs(pos_value) - df["Gross_Exposure"].values)
    if (diff > ATOL_DOLLAR).any():
        idx = int(diff.argmax())
        return False, (
            f"NAV decomposition broken at row {idx}: "
            f"|Gross_NAV-Cash|={abs(pos_value[idx]):.4f} vs "
            f"Gross_Exposure={df['Gross_Exposure'].iloc[idx]:.4f}  "
            f"(diff={diff[idx]:.4f})"
        )

    # Invariant 2 — Cash equation: |ΔCash| == Interval_Turnover (no price dependence)
    cash = df["Cash_Balance"].values
    cash_diff = np.abs(np.diff(cash))
    turnover_t = df["Interval_Turnover"].values[1:]
    diff = np.abs(cash_diff - turnover_t)
    if (diff > ATOL_DOLLAR).any():
        idx_rel = int(diff.argmax())
        idx = idx_rel + 1
        return False, (
            f"Cash equation broken at row {idx}: "
            f"|ΔCash|={cash_diff[idx_rel]:.4f} vs "
            f"Turnover={turnover_t[idx_rel]:.4f}  "
            f"(diff={diff[idx_rel]:.4f})"
        )

    return True, "Pass"


def process_strategy(file_path, strategy_type):
    if not os.path.exists(file_path):
        return False, 0.0, "File not found."

    try:
        df = pd.read_csv(file_path)
    except Exception:
        return False, 0.0, "Could not read CSV."

    # New: structural integrity checks (covers row count, schema, invariants)
    ok, msg = validate_structure(df)
    if not ok:
        return False, 0.0, f"Structural: {msg}"

    # margin check (FP-tolerant)
    if (df["Cash_Balance"] < -ATOL_DOLLAR).any():
        return False, 0.0, "Margin Call: Cash balance dropped below 0."

    # capital ceiling check
    max_exposure = df["Gross_Exposure"].max()
    if strategy_type == "LONG_ONLY" and max_exposure > LONG_ONLY_CAP:
        return False, 0.0, "Breached $1M Capital Limit."
    elif strategy_type == "LONG_SHORT" and max_exposure > LONG_SHORT_CAP:
        return False, 0.0, "Breached $2M Capital Limit."

    # Apply 10 bps friction
    df["Friction_Costs"] = df["Interval_Turnover"] * FEE_BPS
    df["Cumulative_Fees"] = df["Friction_Costs"].cumsum()
    df["Net_NAV"] = df["Gross_NAV"] - df["Cumulative_Fees"]

    # Bankruptcy check
    if (df["Net_NAV"] <= 0).any():
        return False, 0.0, "Bankrupt: Strategy wiped out by transaction fees."

    # Sharpe — guard inf from tiny NAV in case it ever slips past the bankruptcy check
    returns = df["Net_NAV"].pct_change().fillna(0)
    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return False, 0.0, "No valid returns to compute Sharpe."

    mean_return = returns.mean()
    std_dev = returns.std()
    if std_dev == 0 or np.isnan(std_dev):
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (mean_return / std_dev) * np.sqrt(INTERVALS_PER_YEAR)

    return True, round(sharpe_ratio, 4), "Pass"


def batch_grade_folder(target_directory):
    print(f"Scanning directory: {target_directory} ...\n")

    search_pattern = os.path.join(target_directory, "*_longonly_results.csv")
    long_only_files = sorted(glob.glob(search_pattern))

    leaderboard_data = []

    if not long_only_files:
        print("No submissions found.")
        return

    for lo_file in long_only_files:
        base_name = os.path.basename(lo_file)
        team_name = base_name.replace("_longonly_results.csv", "")
        ls_file = os.path.join(target_directory, f"{team_name}_longshort_results.csv")

        print(f"Evaluating Team: {team_name.upper()}")

        lo_pass, lo_sharpe, lo_msg = process_strategy(lo_file, "LONG_ONLY")
        print(f"  Long-Only : {'PASS' if lo_pass else 'FAIL'} | {lo_msg}")

        if os.path.exists(ls_file):
            ls_pass, ls_sharpe, ls_msg = process_strategy(ls_file, "LONG_SHORT")
            print(f"  Long-Short: {'PASS' if ls_pass else 'FAIL'} | {ls_msg}")
        else:
            ls_pass, ls_sharpe, ls_msg = False, 0.0, "Missing file."
            print(f"  Long-Short: FAIL | {ls_msg}")

        if lo_pass and ls_pass:
            blended_sharpe = round((lo_sharpe + ls_sharpe) / 2, 4)
            leaderboard_data.append({
                "Team_Name": team_name.upper(),
                "Blended_Sharpe": blended_sharpe,
                "Status": "Valid",
            })
        else:
            leaderboard_data.append({
                "Team_Name": team_name.upper(),
                "Blended_Sharpe": 0.0,
                "Status": "Disqualified",
            })

    if leaderboard_data:
        df_leaderboard = pd.DataFrame(leaderboard_data)
        df_leaderboard = df_leaderboard.sort_values(
            by="Blended_Sharpe", ascending=False
        ).reset_index(drop=True)

        leaderboard_dir = "leaderboard"
        os.makedirs(leaderboard_dir, exist_ok=True)
        output_filename = os.path.join(leaderboard_dir, "moccm_final_leaderboard_modified.csv")
        df_leaderboard.to_csv(output_filename, index=False)

        print("\n" + "=" * 50)
        print("🏆 MODIFIED LEADERBOARD EXPORTED 🏆")
        print("=" * 50)
        print(f"✅ Leaderboard saved to: {output_filename}")


if __name__ == "__main__":
    submission_folder = "submissions"

    if not os.path.exists(submission_folder):
        os.makedirs(submission_folder)
        print(f"Created '{submission_folder}' folder.")
    else:
        batch_grade_folder(submission_folder)
