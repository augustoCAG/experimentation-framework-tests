"""
Version: 1.0
Date Published: 2024-06-05

Design and analysis utilities for Jampp campaign-level A/B experiments
(paired, per-campaign design).

    Design:        check_spend_balance, estimate_variance_scaling,
                   mde_by_campaigns_and_days
    Analysis:      run_experiment_analysis
    Log-t (BETA):  estimate_variance_scaling_logt_beta,
                   mde_by_campaigns_and_days_logt_beta,
                   run_experiment_analysis_logt_beta

Sign convention: negative Δ → treatment better (one log-t exception).
Full documentation (methodology, assumptions, sign convention, log-t,
known issues): experiment_utils.md, in this folder.
"""

# ----------------------------------------------------------------------------------------------
# -------------------------------- Experiment design functions ---------------------------------
# ----------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy import stats
from statsmodels.stats.power import TTestPower


# ---------------------------------------------------------------------------
# Shared statistical helpers
# ---------------------------------------------------------------------------

def _benjamini_hochberg(p_sorted: np.ndarray, alpha: float) -> tuple:
    """BH step-up on ascending p-values; returns (thresholds, rejected mask)."""
    m            = len(p_sorted)
    bh_threshold = (np.arange(1, m + 1) / m) * alpha
    mask         = p_sorted <= bh_threshold
    max_rejected = int(np.max(np.where(mask)[0]) + 1) if mask.any() else 0
    return bh_threshold, np.arange(1, m + 1) <= max_rejected


# ---------------------------------------------------------------------------
# Spend balance / SRM check
# ---------------------------------------------------------------------------
 
def check_spend_balance(
    df: pd.DataFrame,
    treatment_variant_name: str = 'treatment',
    control_variant_name: str = 'base',
    threshold: float = 0.03,
    spike_threshold: float = 0.10,
    min_days: int = 5,
    alpha: float = 0.05,
    verbose: bool = True,
) -> dict:
    """
    Spend Sample Ratio Mismatch check per campaign: systematic imbalance
    (BH-corrected paired t-test on the daily spend difference, plus `threshold`),
    episodic imbalance (days with |daily ratio − 0.5| > `spike_threshold`) and
    low data (< `min_days` valid days). See experiment_utils.md, section 1.

    Parameters
    ----------
    df : campaign_id, date, variant, spend (one row per campaign × variant × date)
    treatment_variant_name, control_variant_name : `variant` values of each arm
    threshold : aggregate imbalance for the systematic flag (0.03 = 3%)
    spike_threshold : per-day imbalance for the episodic flag (0.10)
    min_days : minimum valid paired days (5)
    alpha : significance level of the BH-corrected t-test (0.05)
    verbose : print a diagnostic summary

    Returns
    -------
    dict: 'campaign_level' (DataFrame, one row per campaign with the flags) and
    'summary' (dict of global totals and flag counts).
    """
    required = {"campaign_id", "date", "variant", "spend"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
 
    d = df.copy()
    d["date"]  = pd.to_datetime(d["date"])
    d["spend"] = pd.to_numeric(d["spend"], errors="coerce")
    d = d.dropna(subset=["campaign_id", "date", "spend"])
 
    # Pivot to spend_control / spend_treatment per (campaign_id, date)
    pivot = d.pivot_table(
        index=["campaign_id", "date"],
        columns="variant",
        values="spend",
        aggfunc="sum",
    )
    pivot.columns.name = None
 
    for var in (control_variant_name, treatment_variant_name):
        if var not in pivot.columns:
            raise ValueError(
                f"Variant '{var}' not found in the 'variant' column. "
                f"Found: {list(pivot.columns)}"
            )
 
    pivot = (
        pivot[[control_variant_name, treatment_variant_name]]
        .rename(columns={
            control_variant_name:   "spend_control",
            treatment_variant_name: "spend_treatment",
        })
        .dropna()
        .reset_index()
    )
 
    # Keep only days where both variants have positive spend
    pivot = pivot[
        (pivot["spend_control"] > 0) & (pivot["spend_treatment"] > 0)
    ].copy()
 
    if pivot.empty:
        raise ValueError(
            "No days found where both variants have positive spend. "
            "Check that variant names and spend values are correct."
        )
 
    # Per-day metrics
    pivot["total_spend"]        = pivot["spend_control"] + pivot["spend_treatment"]
    pivot["daily_ratio"]        = pivot["spend_treatment"] / pivot["total_spend"]
    pivot["daily_diff"]         = pivot["spend_treatment"] - pivot["spend_control"]
    pivot["daily_imbalance_pct"] = pivot["daily_diff"] / pivot["spend_control"] * 100
    pivot["is_spike"]           = (pivot["daily_ratio"] - 0.5).abs() > spike_threshold
 
    # --- Per-campaign summary ---
    records = []
    for campaign_id, grp in pivot.groupby("campaign_id"):
        n_days      = len(grp)
        spend_treat = grp["spend_treatment"].sum()
        spend_ctrl  = grp["spend_control"].sum()
        agg_ratio   = spend_treat / (spend_treat + spend_ctrl)
        agg_imb_pct = (spend_treat - spend_ctrl) / spend_ctrl * 100
 
        # Paired t-test: H₀: mean(spend_treat − spend_ctrl) = 0
        if n_days >= 2:
            t_stat, p_val = stats.ttest_1samp(grp["daily_diff"].values, popmean=0)
            t_stat, p_val = float(t_stat), float(p_val)
        else:
            t_stat, p_val = np.nan, np.nan
 
        # Spike detection
        spike_mask  = grp["is_spike"]
        n_spikes    = int(spike_mask.sum())
        spike_dates = sorted(
            grp.loc[spike_mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        )
 
        records.append({
            "campaign_id":       campaign_id,
            "n_valid_days":      n_days,
            "spend_treatment":   round(spend_treat, 2),
            "spend_control":     round(spend_ctrl,  2),
            "agg_ratio":         round(agg_ratio,   4),
            "agg_imbalance_pct": round(agg_imb_pct, 2),
            "t_stat":            round(t_stat, 4) if not np.isnan(t_stat) else np.nan,
            "p_value":           p_val,
            "n_spike_days":      n_spikes,
            "spike_dates":       spike_dates,
            "flag_low_days":     n_days < min_days,
        })
 
    camp = pd.DataFrame(records)
 
    # --- BH FDR correction (same helper as _segment_analysis) ---
    valid_p = camp["p_value"].notna()
    if valid_p.sum() > 0:
        sub = camp.loc[valid_p, "p_value"].sort_values().reset_index()
        _, significant = _benjamini_hochberg(sub["p_value"].values, alpha)
        sub["t_significant"] = significant
        # Map back to original index
        sig_map = sub.set_index("index")["t_significant"].to_dict()
        camp["t_significant"] = camp.index.map(sig_map).fillna(False).astype(bool)
    else:
        camp["t_significant"] = False
 
    # --- Flag columns ---
    camp["flag_systematic"] = (
        camp["t_significant"]
        & (camp["agg_imbalance_pct"].abs() > threshold * 100)
    )
    camp["flag_episodic"] = camp["n_spike_days"] > 0
    camp["flagged"]       = (
        camp["flag_systematic"]
        | camp["flag_episodic"]
        | camp["flag_low_days"]
    )
 
    camp["p_value"] = camp["p_value"].round(4)
 
    # --- Global summary ---
    global_treat = pivot["spend_treatment"].sum()
    global_ctrl  = pivot["spend_control"].sum()
    global_ratio = global_treat / (global_treat + global_ctrl)
    global_imb   = (global_treat - global_ctrl) / global_ctrl * 100
    abs_imb      = camp["agg_imbalance_pct"].abs()
 
    summary = {
        "global_spend_treatment": round(global_treat, 2),
        "global_spend_control":   round(global_ctrl,  2),
        "global_ratio":           round(global_ratio,  4),
        "global_imbalance_pct":   round(global_imb,    2),
        "n_campaigns_total":      len(camp),
        "n_campaigns_flagged":    int(camp["flagged"].sum()),
        "n_flag_systematic":      int(camp["flag_systematic"].sum()),
        "n_flag_episodic":        int(camp["flag_episodic"].sum()),
        "n_flag_low_days":        int(camp["flag_low_days"].sum()),
        "pct_within_1pct":        round((abs_imb <= 1.0).mean() * 100, 1),
        "pct_within_3pct":        round((abs_imb <= 3.0).mean() * 100, 1),
        "pct_within_5pct":        round((abs_imb <= 5.0).mean() * 100, 1),
    }
 
    if verbose:
        sep = "=" * 64
        print(sep)
        print("  SPEND BALANCE CHECK  (SRM)")
        print(sep)
        print(f"\nGLOBAL")
        print(f"  Spend treatment : {global_treat:>14,.2f}")
        print(f"  Spend control   : {global_ctrl:>14,.2f}")
        print(f"  Global ratio    : {global_ratio:.4f}  (expected 0.5000)")
        print(f"  Global imbalance: {global_imb:+.2f}%")
        print(f"\nCAMPAIGN DISTRIBUTION  ({len(camp)} campaigns)")
        print(f"  Within ±1% : {summary['pct_within_1pct']:.1f}%  of campaigns")
        print(f"  Within ±3% : {summary['pct_within_3pct']:.1f}%  of campaigns")
        print(f"  Within ±5% : {summary['pct_within_5pct']:.1f}%  of campaigns")
        print(f"\nFLAGGED  ({summary['n_campaigns_flagged']} campaigns)")
        print(f"  Systematic  (t-test + |agg| > {threshold*100:.0f}%) : "
              f"{summary['n_flag_systematic']}")
        print(f"  Episodic    (≥1 day with |ratio−0.5| > {spike_threshold*100:.0f}%) : "
              f"{summary['n_flag_episodic']}")
        print(f"  Low data    (< {min_days} valid days)               : "
              f"{summary['n_flag_low_days']}")
 
        flagged = camp[camp["flagged"]].sort_values(
            "agg_imbalance_pct", key=abs, ascending=False
        )
        if not flagged.empty:
            print(f"\nFLAGGED CAMPAIGN DETAILS")
            disp_cols = [
                "campaign_id", "n_valid_days", "agg_imbalance_pct",
                "p_value", "t_significant", "n_spike_days",
                "flag_systematic", "flag_episodic", "flag_low_days",
            ]
            print(flagged[disp_cols].to_string(index=False))
        print(sep)
 
    return {"campaign_level": camp, "summary": summary}


def _variance_scaling_fit(
    daily_df,
    metric_calculator,
    campaign_col,
    date_col,
    candidate_days,
    min_windows_per_day_count,
    log_space,
):
    """
    Window loop and power-law fit shared by both `estimate_variance_scaling`
    functions; returns (c, a, r_squared, points, skipped, n_nonpositive_dropped).
    """
    daily_df = daily_df.sort_values([campaign_col, date_col])

    if log_space and "distance_perc" in daily_df.columns and (daily_df["distance_perc"] < 0).any():
        n_neg = int((daily_df["distance_perc"] < 0).sum())
        print(
            f"  Warning: log_space=True and 'distance_perc' has {n_neg} negative "
            "value(s). log() requires strictly positive input - make sure "
            "metric_calculator returns `1 + distance_perc` (or another "
            "positive-shifted version), otherwise those windows will be "
            "silently dropped."
        )

    history_len_per_campaign = daily_df.groupby(campaign_col)[date_col].nunique()
    max_history = history_len_per_campaign.max()

    if candidate_days is None:
        upper = max(4, max_history // 2)
        n_points = int(min(6, max(2, upper - 2)))
        candidate_days = sorted(set(np.linspace(3, upper, n_points).round().astype(int)))

    numeric_cols = [
        col for col in daily_df.columns
        if col not in (campaign_col, date_col) and pd.api.types.is_numeric_dtype(daily_df[col])
    ]
    context_cols = [col for col in daily_df.columns if col not in (campaign_col, date_col) and col not in numeric_cols]

    per_T_variances = {}
    skipped = []
    n_nonpositive_dropped = 0

    for T in candidate_days:
        campaign_vars = []
        for _, grp in daily_df.groupby(campaign_col):
            grp = grp.sort_values(date_col)
            n_windows = len(grp) - T + 1
            if n_windows < min_windows_per_day_count:
                continue

            window_values = []
            for start in range(n_windows):
                window = grp.iloc[start:start + T]
                row = window[numeric_cols].sum().to_dict()
                for col in context_cols:
                    row[col] = window[col].iloc[0]
                val = metric_calculator(row)
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    continue
                if log_space:
                    if val <= 0:
                        n_nonpositive_dropped += 1
                        continue
                    val = np.log(val)
                window_values.append(val)

            if len(window_values) >= min_windows_per_day_count:
                campaign_vars.append(np.var(window_values, ddof=1))

        if campaign_vars:
            per_T_variances[T] = float(np.mean(campaign_vars))
        else:
            skipped.append(T)

    valid_Ts = sorted(per_T_variances.keys())
    if len(valid_Ts) < 2:
        raise ValueError(
            "Not enough valid window lengths with sufficient overlapping windows "
            "to fit a variance-scaling curve. Provide more historical days, more "
            "campaigns, or lower min_windows_per_day_count."
        )

    empirical_var = np.array([per_T_variances[T] for T in valid_Ts])
    log_T = np.log(valid_Ts)
    log_var = np.log(empirical_var)
    slope, intercept = np.polyfit(log_T, log_var, 1)
    a = -slope
    c = np.exp(intercept)

    fitted_var = c * np.array(valid_Ts, dtype=float) ** (-a)
    ss_res = np.sum((empirical_var - fitted_var) ** 2)
    ss_tot = np.sum((empirical_var - empirical_var.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    points = pd.DataFrame({
        "days": valid_Ts,
        "empirical_variance": empirical_var,
        "fitted_variance": fitted_var,
    })
    return c, a, r_squared, points, skipped, n_nonpositive_dropped


def _print_variance_scaling(header, points, c, a, r_squared, skipped, n_nonpositive_dropped=0):
    """Verbose report of a variance-scaling fit (shared by both versions)."""
    print(header)
    print(points.to_string(index=False))
    print(f"  c = {c:.6g}, a = {a:.4f}  (R2 = {r_squared:.3f})")
    if skipped:
        print(f"  Skipped window lengths (too few overlapping windows): {skipped}")
    if n_nonpositive_dropped:
        print(f"  Dropped {n_nonpositive_dropped} non-positive window value(s) "
              "(log undefined).")
    if abs(a - 1) > 0.3:
        print(
            f"  Note: fitted exponent a={a:.2f} deviates notably from the naive "
            "1/T assumption (a=1) - likely autocorrelation/seasonality in the "
            "daily data. Using the fitted value, not the naive assumption."
        )


def estimate_variance_scaling(
    daily_df,
    metric_calculator,
    campaign_col="campaign_id",
    date_col="date",
    candidate_days=None,
    min_windows_per_day_count=5,
    verbose=True,
):
    """
    Fit variance(T) = c · T^-a of a campaign-level metric over sliding T-day
    windows of historical daily data (called by `mde_by_campaigns_and_days`).
    See experiment_utils.md, section 1.

    Parameters
    ----------
    daily_df : one row per campaign per day, with the columns the calculator needs
    metric_calculator : callable(row of summed window columns) -> metric value
    campaign_col, date_col : campaign / date column names
    candidate_days : window lengths T to fit (default: up to half the longest history)
    min_windows_per_day_count : minimum T-day windows per campaign to use a T (5)
    verbose : print the fit

    Returns
    -------
    dict: 'c', 'a', 'r_squared', 'points' (DataFrame), 'skipped_days'.
    """
    c, a, r_squared, points, skipped, _ = _variance_scaling_fit(
        daily_df, metric_calculator, campaign_col, date_col,
        candidate_days, min_windows_per_day_count, log_space=False,
    )

    if verbose:
        _print_variance_scaling(
            "Variance-scaling fit: variance(T) = c * T^-a",
            points, c, a, r_squared, skipped,
        )

    return {"c": c, "a": a, "r_squared": r_squared, "points": points, "skipped_days": skipped}


def _plot_mde_line(df_out, x_col, xlabel, title):
    """Line plot of MDE % against `x_col` (shared by the MDE functions)."""
    plt.figure(figsize=(8, 5))
    plt.plot(df_out[x_col], df_out["mde_pct"], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Measurable relative impact (%)")
    plt.title(title)
    for x, y in zip(df_out[x_col], df_out["mde_pct"]):
        plt.annotate(f"{y:.2f}%", (x, y))
    plt.show()


def _plot_mde_heatmap(effects, campaigns_options, days_options):
    """Heatmap of MDE % over days × campaigns (shared by the MDE functions)."""
    diff_x = np.diff(campaigns_options)[0] / 2 if len(campaigns_options) > 1 else 0.5
    cmap = plt.cm.rainbow
    my_cmap = cmap(np.arange(cmap.N))
    my_cmap[:, -1] = 0.5
    my_cmap = ListedColormap(my_cmap)

    plt.figure(figsize=(12, 8))
    plt.title(
        "Measurable relative impact vs. Experiment duration (campaigns and days)\n"
        "(paired per-campaign design; variance-vs-days fit empirically from historical data)"
    )
    plt.imshow(
        effects, origin="lower", aspect="auto", cmap=my_cmap,
        extent=[
            min(campaigns_options) - diff_x, max(campaigns_options) + diff_x,
            min(days_options) - 0.5, max(days_options) + 0.5,
        ],
    )
    plt.colorbar()
    plt.ylabel("Days")
    plt.xlabel("Campaigns")
    for i, T in enumerate(days_options):
        for j, N in enumerate(campaigns_options):
            plt.annotate(
                f"{effects[i, j]:.2f}%",
                (campaigns_options[j] - diff_x * 0.5, days_options[i] - 0.12),
                color="#000000",
                fontsize=8,
            )
    plt.xticks(list(campaigns_options))
    plt.yticks(list(days_options))
    plt.show()


def _mde_by_mode(
    mde_relative_pct,
    daily_df,
    campaign_col,
    n_campaigns,
    n_days,
    campaigns_options,
    days_options,
    plot,
):
    """
    MDE output (single value, one-axis table or full table, plus plot) built from
    `mde_relative_pct(N, T)`; shared by the MDE functions.
    """
    if campaigns_options is None:
        max_campaigns = daily_df[campaign_col].nunique()
        campaigns_options = sorted(set(np.linspace(5, max_campaigns, 3).round().astype(int)))

    if n_campaigns is not None and n_days is not None:
        result = mde_relative_pct(n_campaigns, n_days)
        print(f"MDE at N={n_campaigns} campaigns, T={n_days} days: {result:.2f}%")
        return result

    if n_days is not None:
        mde_values = [mde_relative_pct(N, n_days) for N in campaigns_options]
        df_out = pd.DataFrame({"n_campaigns": campaigns_options, "mde_pct": mde_values})
        if plot:
            _plot_mde_line(
                df_out, "n_campaigns", "Number of campaigns",
                f"MDE vs. number of campaigns (T={n_days} days)",
            )
        return df_out

    if n_campaigns is not None:
        mde_values = [mde_relative_pct(n_campaigns, T) for T in days_options]
        df_out = pd.DataFrame({"days": days_options, "mde_pct": mde_values})
        if plot:
            _plot_mde_line(
                df_out, "days", "Days",
                f"MDE vs. experiment duration (N={n_campaigns} campaigns)",
            )
        return df_out

    effects = np.zeros((len(days_options), len(campaigns_options)))
    for i, T in enumerate(days_options):
        for j, N in enumerate(campaigns_options):
            effects[i, j] = mde_relative_pct(N, T)

    if plot:
        _plot_mde_heatmap(effects, campaigns_options, days_options)

    return pd.DataFrame(effects, index=days_options, columns=campaigns_options)


def mde_by_campaigns_and_days(
    daily_df,
    metric_calculator,
    value_a,
    n_campaigns=None,
    n_days=None,
    campaigns_options=None,
    days_options=np.arange(7, 28),
    candidate_scaling_days=None,
    historical_rho=0.0,
    alpha=0.05,
    power=0.8,
    campaign_col="campaign_id",
    date_col="date",
    plot=True,
):
    """
    Relative MDE (%) of the paired per-campaign design for N campaigns and T
    days. Fix `n_campaigns` and/or `n_days` for a single value or a one-axis
    table; leave both free for the full table and heatmap. Assumptions and
    details: experiment_utils.md, section 1.

    Parameters
    ----------
    daily_df, metric_calculator : historical data, passed to estimate_variance_scaling
    value_a : baseline mean of the metric (turns the MDE into a relative %)
    n_campaigns, n_days : fix one or both axes
    campaigns_options, days_options : values shown on the free axis / axes
    candidate_scaling_days : window lengths used to fit the variance curve
    historical_rho : CUPED correlation (0.0 = no CUPED benefit)
    alpha, power : t-test significance level and power
    campaign_col, date_col : campaign / date column names
    plot : draw the heatmap / line plot

    Returns
    -------
    float (both axes fixed) or pd.DataFrame (MDE % table).
    """
    scaling = estimate_variance_scaling(
        daily_df,
        metric_calculator,
        campaign_col=campaign_col,
        date_col=date_col,
        candidate_days=candidate_scaling_days,
        verbose=True,
    )
    c, a = scaling["c"], scaling["a"]

    # Calculate the std dev of our metric
    def sigma_delta(T):
        return np.sqrt(c * T ** (-a))

    # Reduce the std dev with CUPED
    def sigma_delta_cuped(T):
        return sigma_delta(T) * np.sqrt(max(0.0, 1 - historical_rho ** 2))

    # d (Cohen's d) * sigma = smallest detectable absolute difference at power/alpha/N;
    # / value_a * 100 turns it into a relative % MDE
    def mde_relative_pct(N, T):
        d = TTestPower().solve_power(effect_size=None, nobs=N, alpha=alpha, power=power, alternative="two-sided")
        return d * sigma_delta_cuped(T) / value_a * 100

    return _mde_by_mode(
        mde_relative_pct, daily_df, campaign_col, n_campaigns, n_days,
        campaigns_options, days_options, plot,
    )



# ----------------------------------------------------------------------------------------------
# ----------------------------- Post Experiment (impact) functions -----------------------------
# ----------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROAS_GOAL_TYPES = {"ROAS"}

# Numerator / denominator columns used to recompute CPx per goal type.
_CPX_COLS: dict[str, tuple[str, str]] = {
    "CPI":  ("spend", "installs"),
    "CPA":  ("spend", "goal_event"),
    "CPC":  ("spend", "clicks"),
    "ROAS": ("goal_event_value", "spend"),   # ROAS = revenue / spend
}

# Goal type / value columns per `goal_type_source` ('contractual' = agreed goal,
# 'operative' = goal applied operationally); passed to helpers as goal_type_col / goal_value_col.
_GOAL_COLS: dict[str, tuple[str, str]] = {
    "contractual": ("contractual_goal_type", "contractual_goal_value"),
    "operative":   ("operative_goal_type",   "operational_goal_value"),
}

SEGMENT_COLS_BASE = ["campaign_type"]


# ---------------------------------------------------------------------------
# 1. Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(
    df: pd.DataFrame,
    metric_col: str = "distance_perc",
    sign_convention: str = "lower_is_better",
    goal_value_col: str = "contractual_goal_value",
) -> pd.DataFrame:
    """
    Coerce types and build `metric_analysis`. For distance_perc / absolute_cpx it is
    the raw column (the ROAS flip is applied later, in `_build_campaign_table`; the
    daily variance is sign-invariant); custom metrics are negated if 'higher_is_better'.
    """
    if metric_col not in df.columns:
        raise ValueError(
            f"metric_col='{metric_col}' not found in the DataFrame. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    df = df.copy()

    # Coerce standard numeric columns that exist in the DataFrame
    _std_numerics = [
        "bids", "impressions", "spend", "clicks", "installs", "goal_event",
        "goal_event_value", goal_value_col, "distance_perc", "absolute_cpx"
    ]
    for col in _std_numerics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Coerce the metric column if it is not in the standard list
    if metric_col not in _std_numerics:
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], utc=True)

    if metric_col in ("distance_perc", "absolute_cpx"):
        # No sign change here: the ROAS flip happens once, in `_build_campaign_table`
        df["metric_analysis"] = df[metric_col]
    else:
        # Global sign based on user-specified convention
        sign = -1 if sign_convention == "higher_is_better" else 1
        df["metric_analysis"] = sign * df[metric_col].astype(float)

    return df


# ---------------------------------------------------------------------------
# 2. Aggregate CPx → % distance per campaign × variant
# ---------------------------------------------------------------------------

def _aggregate_distance(
    df: pd.DataFrame,
    metric_: str = 'distance_perc',
    goal_type_col: str = "contractual_goal_type",
    goal_value_col: str = "contractual_goal_value",
) -> pd.DataFrame:
    """
    Ratio-of-sums CPx (and % distance to goal) per campaign × variant (no sign
    change: ROAS is flipped in `_build_campaign_table`); rows with a zero
    denominator are dropped.
    """
    goal_types = df[goal_type_col].unique()
    parts = []

    for goal_type in goal_types:
        subset = df[df[goal_type_col] == goal_type].copy()
        num_col, den_col = _CPX_COLS.get(goal_type, ("spend", "goal_event"))

        agg = (
            subset.groupby(
                ["campaign_id", "variant", "campaign_type",
                 goal_type_col, goal_value_col],
                as_index=False,
            )
            .agg(
                num_sum=(num_col,  "sum"),
                den_sum=(den_col,  "sum"),
                spend_sum=("spend", "sum"),   # always track spend for weighting
            )
        )

        # Drop rows where the metric denominator is zero
        agg = agg[agg["den_sum"] > 0].copy()

        agg["cpx_agg"]  = agg["num_sum"] / agg["den_sum"]
        if metric_ == 'distance_perc':
            agg["dist_agg"] = agg["cpx_agg"] / agg[goal_value_col] - 1
        elif metric_ == 'absolute_cpx':
            agg["dist_agg"] = agg["cpx_agg"]

        # Output column always named `metric_value` (raw direction)
        agg["metric_value"] = agg["dist_agg"]

        parts.append(agg)

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 2b. Window-level custom metric supplied by the caller (agg_df)  [shared with log-t]
# ---------------------------------------------------------------------------

def _prepare_agg_df(
    agg_df: Optional[pd.DataFrame],
    df: pd.DataFrame,
    sign_convention: str,
    control_variant_name: str,
    treatment_variant_name: str,
    use_spend_weights: bool,
    goal_type_col: str = "contractual_goal_type",
    goal_value_col: str = "contractual_goal_value",
    log_space: bool = False,
) -> pd.DataFrame:
    """
    Validate the user-supplied window-level custom metric (`agg_df`, one row per
    campaign x variant, in the metric's own direction) and return it in the same
    format as `_aggregate_distance` (columns campaign_id, variant, metric_value,
    spend_sum and, when available, campaign_type / goal columns).
    `sign_convention` is applied like in the daily path: raw -> -x, log -> 1/x
    (only where x > 0, so the log-t positivity filter still catches the rest).
    """
    if agg_df is None:
        raise ValueError(
            "agg_df is required for custom metrics: one row per campaign x variant "
            "with columns campaign_id, variant, metric_value (window-level value, in "
            "the metric's own direction)."
        )

    required = ["campaign_id", "variant", "metric_value"]
    if use_spend_weights:
        required.append("spend_sum")
    missing = [c for c in required if c not in agg_df.columns]
    if missing:
        raise ValueError(
            f"agg_df is missing required column(s) {missing}. "
            f"Available columns: {sorted(agg_df.columns.tolist())}"
        )

    agg = agg_df.copy()

    if agg.duplicated(["campaign_id", "variant"]).any():
        n_dup = int(agg.duplicated(["campaign_id", "variant"], keep=False).sum())
        raise ValueError(
            f"agg_df must have one row per campaign x variant; {n_dup} row(s) are duplicated."
        )

    valid_variants = {control_variant_name, treatment_variant_name}
    bad_variants = sorted(set(agg["variant"].unique()) - valid_variants, key=str)
    if bad_variants:
        raise ValueError(
            f"agg_df has variant name(s) {bad_variants} that do not match "
            f"control_variant_name='{control_variant_name}' / "
            f"treatment_variant_name='{treatment_variant_name}'."
        )

    df_campaigns  = set(df["campaign_id"].unique())
    agg_campaigns = set(agg["campaign_id"].unique())

    missing_in_agg = df_campaigns - agg_campaigns
    if missing_in_agg:
        raise ValueError(
            f"{len(missing_in_agg)} campaign(s) in df are missing from agg_df: "
            f"{sorted(missing_in_agg, key=str)[:10]}"
            f"{' ...' if len(missing_in_agg) > 10 else ''}"
        )

    extra_in_agg = agg_campaigns - df_campaigns
    if extra_in_agg:
        warnings.warn(
            f"{len(extra_in_agg)} campaign(s) in agg_df are missing from df and will "
            f"be ignored: {sorted(extra_in_agg, key=str)[:10]}"
            f"{' ...' if len(extra_in_agg) > 10 else ''}",
            UserWarning,
        )
        agg = agg[agg["campaign_id"].isin(df_campaigns)].copy()

    # Sign convention, same as the daily series of each path
    agg["metric_value"] = pd.to_numeric(agg["metric_value"], errors="coerce").astype(float)
    if sign_convention == "higher_is_better":
        if log_space:
            positive = agg["metric_value"] > 0
            agg.loc[positive, "metric_value"] = 1 / agg.loc[positive, "metric_value"]
        else:
            agg["metric_value"] = -agg["metric_value"]

    if use_spend_weights:
        agg["spend_sum"] = pd.to_numeric(agg["spend_sum"], errors="coerce")
    else:
        agg["spend_sum"] = np.nan

    # Segment columns: warn and skip the analysis if absent from agg_df
    for seg_name, seg_cols in (
        ("campaign_type", ["campaign_type"]),
        (goal_type_col, [goal_type_col, goal_value_col]),
    ):
        absent = [c for c in seg_cols if c not in agg.columns]
        if absent:
            warnings.warn(
                f"agg_df has no {absent} column(s): the '{seg_name}' segment analysis "
                "will be skipped.",
                UserWarning,
            )

    # Campaign attributes must be one value per campaign (they are pivot keys)
    ctx_cols = [c for c in ["campaign_type", goal_type_col, goal_value_col] if c in agg.columns]
    if ctx_cols:
        varying = agg.groupby("campaign_id")[ctx_cols].nunique(dropna=False).max()
        varying = varying[varying > 1].index.tolist()
        if varying:
            warnings.warn(
                f"agg_df column(s) {varying} differ between variants of the same "
                "campaign; the first value per campaign is used.",
                UserWarning,
            )
            agg[ctx_cols] = agg.groupby("campaign_id")[ctx_cols].transform("first")

    keep = ["campaign_id", "variant", "metric_value", "spend_sum"] + ctx_cols
    return agg[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Daily paired differences (for variance estimation)
# ---------------------------------------------------------------------------

def _daily_pairs(
    df: pd.DataFrame,
    lift_type: str = "absolute",
    treatment_variant_name: str = "treatment", 
    control_variant_name: str = "base"
) -> pd.DataFrame:
    """
    Daily treatment − control differences of `metric_analysis` (absolute, or
    relative to |control|), used for the variance estimate.
    """
    variants = df[df["variant"].isin([control_variant_name, treatment_variant_name])].copy()

    pivot = variants.pivot_table(
        index=["campaign_id", "date"],
        columns="variant",
        values="metric_analysis",
        aggfunc="mean",   # mean handles multiple rows per (campaign, date, variant)
    ).reset_index()
    pivot.columns.name = None

    pivot = pivot.dropna(subset=[control_variant_name, treatment_variant_name])

    if lift_type == "relative":
        denom = pivot[control_variant_name].abs()
        pivot = pivot[denom > 0].copy()          # drop days where control = 0
        denom = pivot[control_variant_name].abs()
        pivot["daily_delta"] = (
            (pivot[treatment_variant_name] - pivot[control_variant_name]) / denom
        )
    else:
        pivot["daily_delta"] = pivot[treatment_variant_name] - pivot[control_variant_name]

    return pivot[["campaign_id", "date", "daily_delta"]]


# ---------------------------------------------------------------------------
# 4. Campaign-level pairing and filtering
# ---------------------------------------------------------------------------

def _attach_total_spend(
    wide: pd.DataFrame,
    agg_metric: pd.DataFrame,
    treatment_variant_name: str,
    control_variant_name: str,
) -> pd.DataFrame:
    """
    Add `total_spend_c` (control + treatment spend) for spend weighting; NaN when
    `agg_metric` has no 'spend_sum'.
    """
    # Total spend per campaign (both variants combined) — used for spend weighting
    if "spend_sum" in agg_metric.columns and agg_metric["spend_sum"].notna().any():
        spend_wide = agg_metric.pivot_table(
            index=["campaign_id"],
            columns="variant",
            values="spend_sum",
            aggfunc="first",
        ).reset_index()
        spend_wide.columns.name = None
        spend_wide["total_spend_c"] = (
            spend_wide.get(control_variant_name, 0).fillna(0)
            + spend_wide.get(treatment_variant_name, 0).fillna(0)
        )
        wide = wide.merge(spend_wide[["campaign_id", "total_spend_c"]],
                          on="campaign_id", how="left")
    else:
        wide["total_spend_c"] = np.nan
    return wide


def _attach_variance_and_filter(
    wide: pd.DataFrame,
    daily_pair_df: pd.DataFrame,
    delta_col: str,
    var_col: str,
    min_valid_days: int,
) -> pd.DataFrame:
    """
    Add Var(daily delta) / n_valid_days and drop campaigns with < `min_valid_days`
    days or a non-positive variance (with a warning).
    """
    var_df = (
        daily_pair_df.groupby("campaign_id")
        .agg(
            n_valid_days=(delta_col, "count"),
            var_daily=(delta_col, lambda x: x.var(ddof=1)),
        )
        .reset_index()
    )
    var_df[var_col] = var_df["var_daily"] / var_df["n_valid_days"]

    camp = wide.merge(var_df, on="campaign_id", how="inner")

    before = len(camp)
    camp   = camp[camp["n_valid_days"] >= min_valid_days].copy()
    camp   = camp.dropna(subset=[var_col])
    camp   = camp[camp[var_col] > 0].copy()
    after  = len(camp)

    if after < before:
        warnings.warn(
            f"{before - after} campaign(s) excluded (< {min_valid_days} valid days "
            "or undefined variance).",
            UserWarning,
        )

    return camp.reset_index(drop=True)


def _build_campaign_table(
    agg_metric: pd.DataFrame,
    daily_pair_df: pd.DataFrame,
    min_valid_days: int,
    lift_type: str = "absolute",
    treatment_variant_name: str = "treatment",
    control_variant_name: str = "base",
    goal_type_col: str = "contractual_goal_type",
    goal_value_col: str = "contractual_goal_value",
    flip_roas: bool = False,
) -> pd.DataFrame:
    """
    Campaign table: window-level δ_c (absolute or relative), its daily-based
    variance and total spend; applies the min-days / variance filter.
    `flip_roas` negates the control / treatment window values of ROAS rows (higher
    ROAS = better → negative Δ = better); the single ROAS flip of the raw path.
    """
    # Determine which index columns are present (varies by metric type)
    _possible_idx = [
        "campaign_id", "campaign_type", goal_type_col, goal_value_col,
    ]
    index_cols = [c for c in _possible_idx if c in agg_metric.columns]

    wide = agg_metric.pivot_table(
        index=index_cols,
        columns="variant",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    wide = wide.dropna(subset=[control_variant_name, treatment_variant_name])

    if flip_roas and goal_type_col in wide.columns:
        is_roas = wide[goal_type_col].isin(ROAS_GOAL_TYPES)
        wide.loc[is_roas, [control_variant_name, treatment_variant_name]] *= -1

    if lift_type == "relative":
        denom = wide[control_variant_name].abs()
        # Campaigns where control window value is zero: metric undefined
        valid = denom > 0
        wide  = wide[valid].copy()
        denom = wide[control_variant_name].abs()
        wide["delta_c"] = (
            (wide[treatment_variant_name] - wide[control_variant_name]) / denom
        )
    else:
        wide["delta_c"] = wide[treatment_variant_name] - wide[control_variant_name]

    wide = wide.rename(
        columns={
            control_variant_name:   "metric_control",
            treatment_variant_name: "metric_treatment",
        }
    )

    wide = _attach_total_spend(wide, agg_metric, treatment_variant_name, control_variant_name)

    return _attach_variance_and_filter(
        wide, daily_pair_df, "daily_delta", "var_delta_c", min_valid_days,
    )


# ---------------------------------------------------------------------------
# 5. CUPED
# ---------------------------------------------------------------------------

def _cuped_adjust(
    X: np.ndarray,
    Y: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> tuple:
    """
    CUPED core: returns (Y − θ·(X − X̄), θ, variance reduction), weighted when
    `weights` is given.
    """
    if weights is not None:
        w      = weights / weights.sum()                       # normalise to sum-to-1
        X_bar  = np.dot(w, X)
        Y_bar  = np.dot(w, Y)
        cov_YX = np.dot(w, (Y - Y_bar) * (X - X_bar))
        var_X  = np.dot(w, (X - X_bar) ** 2)
        theta  = cov_YX / var_X if var_X > 0 else 0.0
        cuped_vals = Y - theta * (X - X_bar)
        # Weighted variance reduction
        var_before = np.dot(w, (Y - Y_bar) ** 2)
        var_after  = np.dot(w, (cuped_vals - np.dot(w, cuped_vals)) ** 2)
    else:
        cov_matrix = np.cov(Y, X, ddof=1)
        theta      = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0.0
        cuped_vals = Y - theta * (X - X.mean())
        var_before = np.var(Y, ddof=1)
        var_after  = np.var(cuped_vals, ddof=1)

    reduction = 1 - var_after / var_before if var_before > 0 else 0.0
    return cuped_vals, theta, reduction


def _apply_cuped(
    camp: pd.DataFrame,
    pre_experiment_dist: Optional[pd.Series],
    random_seed: int,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    CUPED on δ_c with the pre-experiment covariate (synthetic if None, in-experiment
    fallback for missing campaigns); stores θ and the reduction in `.attrs`.
    """
    df  = camp.copy()
    rng = np.random.default_rng(random_seed)

    if not isinstance(pre_experiment_dist, pd.Series):
        warnings.warn(
            "No pre-experiment data provided — using a synthetic CUPED covariate. "
            "Pass a real pd.Series (indexed by campaign_id) via `pre_experiment_dist` "
            "before using results in production.",
            UserWarning,
        )
        baseline  = (df["metric_control"] + df["metric_treatment"]) / 2
        noise_sd  = baseline.std(ddof=1) * 0.8 if baseline.std(ddof=1) > 0 else 0.1
        synthetic = baseline + rng.normal(0, noise_sd, size=len(df))
        df["pre_exp_dist"] = synthetic.values
    else:
        df["pre_exp_dist"] = df["campaign_id"].map(pre_experiment_dist)
        missing = df["pre_exp_dist"].isna().sum()
        if missing > 0:
            warnings.warn(
                f"{missing} campaign(s) have no pre-experiment covariate; "
                "falling back to in-experiment average.",
                UserWarning,
            )
            fallback = (df["metric_control"] + df["metric_treatment"]) / 2
            df["pre_exp_dist"] = df["pre_exp_dist"].fillna(fallback)

    X = df["pre_exp_dist"].values.astype(float)
    Y = df["delta_c"].values.astype(float)

    df["delta_c_cuped"], theta, reduction = _cuped_adjust(X, Y, weights)

    df.attrs["cuped_theta"]              = theta
    df.attrs["cuped_variance_reduction"] = reduction

    return df


# ---------------------------------------------------------------------------
# 6. Random-effects meta-analysis (DerSimonian-Laird)  [shared with log-t]
# ---------------------------------------------------------------------------

def _random_effects_meta(delta: np.ndarray, sigma2: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pooling of δ_c (inverse-variance weights)."""
    k  = len(delta)
    w  = 1.0 / sigma2

    delta_fe = np.dot(w, delta) / w.sum()
    Q        = np.dot(w, (delta - delta_fe) ** 2)

    c    = w.sum() - np.dot(w, w) / w.sum()
    tau2 = max(0.0, (Q - (k - 1)) / c)

    w_re   = 1.0 / (sigma2 + tau2)
    pooled = np.dot(w_re, delta) / w_re.sum()
    se     = np.sqrt(1.0 / w_re.sum())
    I2     = max(0.0, (Q - (k - 1)) / Q) * 100 if Q > 0 else 0.0

    z_crit = stats.norm.ppf(0.975)
    return {
        "pooled_estimate": pooled,
        "se":              se,
        "ci_lower":        pooled - z_crit * se,
        "ci_upper":        pooled + z_crit * se,
        "tau2":            tau2,
        "I2":              I2,
        "Q":               Q,
        "Q_pvalue":        1 - stats.chi2.cdf(Q, df=k - 1),
        "k":               k,
    }


# ---------------------------------------------------------------------------
# 7. Hypothesis test  [shared with log-t]
# ---------------------------------------------------------------------------

def _hypothesis_test(delta_cuped: np.ndarray, alpha: float) -> dict:
    """One-sample t-test of the CUPED-adjusted δ_c (H₀: mean = 0)."""
    n         = len(delta_cuped)
    t_stat, p = stats.ttest_1samp(delta_cuped, popmean=0)
    se        = np.std(delta_cuped, ddof=1) / np.sqrt(n)
    mean_d    = np.mean(delta_cuped)
    t_crit    = stats.t.ppf(1 - alpha / 2, df=n - 1)

    return {
        "mean_delta":  mean_d,
        "se":          se,
        "ci_lower":    mean_d - t_crit * se,
        "ci_upper":    mean_d + t_crit * se,
        "t_stat":      t_stat,
        "p_value":     p,
        "significant": bool(p < alpha),
        "direction":   "treatment_better" if mean_d < 0 else "control_better",
        "n_campaigns": n,
        "alpha":       alpha,
    }


# ---------------------------------------------------------------------------
# 7b. Spend-weighted hypothesis test (bootstrap)  [shared with log-t]
# ---------------------------------------------------------------------------

def _weighted_hypothesis_test(
    delta_cuped: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    n_boot: int = 2000,
    random_seed: int = 42,
) -> dict:
    """Spend-weighted mean of δ_c, with a bootstrap SE and normal approximation."""
    rng = np.random.default_rng(random_seed)
    n   = len(delta_cuped)
    w   = weights / weights.sum()

    mean_d = float(np.dot(w, delta_cuped))

    # Bootstrap: resample campaigns with replacement
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx       = rng.integers(0, n, size=n)
        w_b       = weights[idx]
        boot_means[i] = np.dot(w_b / w_b.sum(), delta_cuped[idx])

    se     = float(np.std(boot_means, ddof=1))
    z_stat = mean_d / se if se > 0 else 0.0
    p      = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    return {
        "mean_delta":   mean_d,
        "se":           se,
        "ci_lower":     mean_d - z_crit * se,
        "ci_upper":     mean_d + z_crit * se,
        "z_stat":       z_stat,
        "p_value":      p,
        "significant":  bool(p < alpha),
        "direction":    "treatment_better" if mean_d < 0 else "control_better",
        "n_campaigns":  n,
        "alpha":        alpha,
        "n_bootstrap":  n_boot,
    }


# ---------------------------------------------------------------------------
# 7c. Sign-flip permutation test  [shared with log-t]
# ---------------------------------------------------------------------------

def _permutation_test(
    delta_cuped: np.ndarray,
    alpha: float,
    weights: Optional[np.ndarray] = None,
    n_perm: int = 10_000,
    random_seed: int = 42,
) -> dict:
    """Sign-flip permutation test of the (optionally spend-weighted) mean δ_c."""
    rng = np.random.default_rng(random_seed)
    n   = len(delta_cuped)

    if weights is not None:
        w        = (weights / weights.sum()).astype(float)
        observed = float(np.dot(w, delta_cuped))
    else:
        w        = None
        observed = float(np.mean(delta_cuped))

    # Vectorised null distribution: generate all n_perm × n sign matrices at once
    signs_matrix = rng.integers(0, 2, size=(n_perm, n)) * 2 - 1     # {-1, +1}
    perm_data    = delta_cuped * signs_matrix                         # broadcast (n_perm, n)
    if w is not None:
        perm_stats = perm_data @ w          # (n_perm,) spend-weighted means
    else:
        perm_stats = perm_data.mean(axis=1) # (n_perm,) equal-weighted means

    # Two-sided p-value: fraction of null statistics at least as extreme as observed
    p = float(np.mean(np.abs(perm_stats) >= np.abs(observed)))
    p = max(p, 1.0 / n_perm)   # floor at 1/B to avoid p = 0

    # CI: analytical SE kept consistent with the corresponding parametric test
    # so CIs remain directly comparable regardless of which test is chosen.
    if w is not None:
        # Weighted case: SE from the permutation null distribution
        se     = float(np.std(perm_stats, ddof=1))
        z_crit = stats.norm.ppf(1 - alpha / 2)
        ci_lo  = observed - z_crit * se
        ci_hi  = observed + z_crit * se
    else:
        # Equal-weighted: same SE formula as t-test (permutation SE ≈ identical)
        se     = float(np.std(delta_cuped, ddof=1) / np.sqrt(n))
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        ci_lo  = observed - t_crit * se
        ci_hi  = observed + t_crit * se

    return {
        "mean_delta":     observed,
        "se":             se,
        "ci_lower":       ci_lo,
        "ci_upper":       ci_hi,
        "p_value":        p,
        "significant":    bool(p < alpha),
        "direction":      "treatment_better" if observed < 0 else "control_better",
        "n_campaigns":    n,
        "alpha":          alpha,
        "n_permutations": n_perm,
    }


# ---------------------------------------------------------------------------
# 8. Segment analysis with Benjamini-Hochberg correction  [shared with log-t]
# ---------------------------------------------------------------------------

def _segment_analysis(
    camp: pd.DataFrame,
    segment_col: str,
    alpha: float,
    test_method: str = "t-test",
    n_perm: int = 10_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Per-segment test of `delta_c_cuped` with BH FDR across the segments of
    `segment_col` (segments with < 3 campaigns are skipped).
    """
    records = []
    for seg, grp in camp.groupby(segment_col):
        delta = grp["delta_c_cuped"].values
        if len(delta) < 3:
            continue

        if test_method == "permutation":
            res = _permutation_test(
                delta, alpha, n_perm=n_perm, random_seed=random_seed
            )
            records.append(
                {
                    "segment":     seg,
                    "n_campaigns": len(delta),
                    "mean_delta":  res["mean_delta"],
                    "se":          res["se"],
                    "p_value_raw": res["p_value"],
                    "direction":   res["direction"],
                }
            )
        else:
            t_stat, p = stats.ttest_1samp(delta, popmean=0)
            se        = np.std(delta, ddof=1) / np.sqrt(len(delta))
            records.append(
                {
                    "segment":     seg,
                    "n_campaigns": len(delta),
                    "mean_delta":  float(np.mean(delta)),
                    "se":          float(se),
                    "t_stat":      float(t_stat),
                    "p_value_raw": float(p),
                    "direction":   "treatment_better" if np.mean(delta) < 0 else "control_better",
                }
            )

    if not records:
        return pd.DataFrame()

    out = pd.DataFrame(records).sort_values("p_value_raw").reset_index(drop=True)

    bh_threshold, significant = _benjamini_hochberg(out["p_value_raw"].values, alpha)
    out["bh_threshold"]   = bh_threshold
    out["significant_bh"] = significant

    return out


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def _test_label_and_stat_line(
    overall: dict,
    alpha: float,
    use_spend_weights: bool,
    test_method: str,
) -> tuple:
    """Header and optional statistic line of the overall-test block (shared)."""
    is_perm = test_method == "permutation"
    if use_spend_weights:
        if is_perm:
            test_label = (
                f"OVERALL HYPOTHESIS TEST — spend-weighted  "
                f"(α = {alpha}, permutation n={overall['n_permutations']:,})"
            )
            stat_line = None
        else:
            test_label = (
                f"OVERALL HYPOTHESIS TEST — spend-weighted  "
                f"(α = {alpha}, bootstrap n={overall['n_bootstrap']})"
            )
            stat_line = f"  z-statistic (boot.)  : {overall['z_stat']:+.4f}"
    else:
        if is_perm:
            test_label = (
                f"OVERALL HYPOTHESIS TEST — equal-weighted  "
                f"(α = {alpha}, permutation n={overall['n_permutations']:,})"
            )
            stat_line = None
        else:
            test_label = f"OVERALL HYPOTHESIS TEST — equal-weighted  (α = {alpha})"
            stat_line  = f"  t-statistic         : {overall['t_stat']:+.4f}"
    return test_label, stat_line


def _print_caveats(use_spend_weights: bool, alpha: float) -> None:
    """Closing caveat lines of the summary (shared by both printers)."""
    print(f"\n  ⚠  Segment breakdowns always use equal weights (exploratory).")
    if use_spend_weights:
        print(f"  ⚠  Overall test is spend-weighted; meta-analysis uses")
        print(f"     inverse-variance weights (these answer different questions).")
    print(f"  ⚠  Peeking caveat: if results were monitored mid-experiment,")
    print(f"     the nominal α={alpha} is anti-conservative.")


def _print_summary(
    overall: dict,
    meta: dict,
    cuped_stats: dict,
    segments: dict,
    n_raw: int,
    n_excluded: int,
    alpha: float,
    use_spend_weights: bool,
    test_method: str = "t-test",
    metric_col: str = "distance_perc",
    sign_convention: str = "lower_is_better",
    lift_type: str = "absolute",
) -> None:
    sep = "=" * 64

    print(sep)
    print("  POST-EXPERIMENT ANALYSIS — JAMPP A/B TEST")
    print(sep)

    weight_mode = "spend-weighted" if use_spend_weights else "equal-weighted"
    delta_unit  = "relative (fractional)" if lift_type == "relative" else "absolute"
    print(f"\nCAMPAIGNS")
    print(f"  In dataset          : {n_raw}")
    print(f"  Included in analysis: {overall['n_campaigns']}")
    print(f"  Excluded (low data) : {n_excluded}")
    print(f"  Metric              : {metric_col}  ({sign_convention}, {delta_unit} Δ)")
    print(f"  Weighting mode      : {weight_mode}")
    print(f"  Hypothesis test     : {test_method}")

    print(f"\nCUPED VARIANCE REDUCTION")
    print(f"  θ (covariate weight): {cuped_stats['theta']:+.4f}")
    print(f"  Variance reduced by : {cuped_stats['variance_reduction_pct']:.1f}%")
    print(f"  Covariate source    : {cuped_stats['note']}")

    sig_label = "✓  SIGNIFICANT" if overall["significant"] else "✗  NOT SIGNIFICANT"
    dir_label = overall["direction"].replace("_", " ").upper()

    # Build test label and optional inline stat line
    test_label, stat_line = _test_label_and_stat_line(overall, alpha, use_spend_weights, test_method)

    print(f"\n{test_label}")
    print(f"  {sig_label}  |  {dir_label}")
    print(f"  Mean Δ (CUPED-adj.) : {overall['mean_delta']:+.4f}")
    print(f"  95% CI              : [{overall['ci_lower']:+.4f},  {overall['ci_upper']:+.4f}]")
    if stat_line:
        print(stat_line)
    print(f"  p-value             : {overall['p_value']:.4f}")
    print(f"  N campaigns         : {overall['n_campaigns']}")

    print(f"\nRANDOM-EFFECTS META-ANALYSIS  (DerSimonian-Laird, inverse-variance weighted)")
    print(f"  Pooled Δ            : {meta['pooled_estimate']:+.4f}  (SE = {meta['se']:.4f})")
    print(f"  95% CI              : [{meta['ci_lower']:+.4f},  {meta['ci_upper']:+.4f}]")
    print(f"  τ² (heterogeneity)  : {meta['tau2']:.4f}")
    print(f"  I²                  : {meta['I2']:.1f}%")
    print(f"  Q statistic         : {meta['Q']:.2f}  (p = {meta['Q_pvalue']:.3f})")

    for col, seg_df in segments.items():
        if seg_df.empty:
            continue
        print(f"\nSEGMENT BREAKDOWN — {col.upper()}  (equal-weighted, BH FDR @ α={alpha})")
        hdr = f"  {'Segment':<28} {'N':>4}  {'Mean Δ':>8}  {'p-value':>8}  {'BH-sig':>7}  Direction"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for _, row in seg_df.iterrows():
            sig  = "Yes *" if row["significant_bh"] else "No"
            dirn = row["direction"].replace("_", " ")
            print(
                f"  {str(row['segment']):<28} {int(row['n_campaigns']):>4}"
                f"  {row['mean_delta']:>+8.4f}  {row['p_value_raw']:>8.4f}"
                f"  {sig:>7}  {dirn}"
            )

    print(f"\n{sep}")
    print("  SIGN CONVENTION")
    print("  Negative Δ → treatment performs BETTER than control")
    print("  Positive Δ → treatment performs WORSE  than control")
    if metric_col == "distance_perc":
        print("  (Cost metrics kept as-is: negative = below goal = better.")
        print("   ROAS flipped: negative = above goal = better.)")
    else:
        if sign_convention == "higher_is_better":
            print(f"  ({metric_col} is higher-is-better; values were flipped internally.)")
        else:
            print(f"  ({metric_col} is lower-is-better; no sign change applied.)")
    if lift_type == "relative":
        print("  Δ is fractional: (treatment − control) / |control|.")
        print("  Interpret as % change relative to control.")
    _print_caveats(use_spend_weights, alpha)
    print(sep)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_VALID_CONVENTIONS = {"lower_is_better", "higher_is_better"}
_VALID_LIFTS       = {"absolute", "relative"}
_VALID_TESTS       = {"t-test", "permutation"}
_NO_LIFT_TYPE      = object()   # sentinel: the log-space pipeline has no `lift_type`


def _validate_inputs(
    df: pd.DataFrame,
    sign_convention: str,
    test_method: str,
    goal_type_source: str,
    lift_type=_NO_LIFT_TYPE,
) -> tuple:
    """
    Validate the run functions' shared arguments; return (goal_type_col, goal_value_col).
    """
    if sign_convention not in _VALID_CONVENTIONS:
        raise ValueError(
            f"sign_convention must be one of {_VALID_CONVENTIONS}, "
            f"got '{sign_convention}'."
        )
    if lift_type is not _NO_LIFT_TYPE and lift_type not in _VALID_LIFTS:
        raise ValueError(
            f"lift_type must be one of {_VALID_LIFTS}, got '{lift_type}'."
        )
    if test_method not in _VALID_TESTS:
        raise ValueError(
            f"test_method must be one of {_VALID_TESTS}, got '{test_method}'."
        )
    if goal_type_source not in _GOAL_COLS:
        raise ValueError(
            f"goal_type_source must be one of {sorted(_GOAL_COLS)}, "
            f"got '{goal_type_source}'."
        )

    goal_type_col, goal_value_col = _GOAL_COLS[goal_type_source]
    missing_goal_cols = {goal_type_col, goal_value_col} - set(df.columns)
    if missing_goal_cols:
        raise ValueError(
            f"goal_type_source='{goal_type_source}' requires column(s) "
            f"{sorted(missing_goal_cols)}, not found in the DataFrame. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )
    return goal_type_col, goal_value_col


def run_experiment_analysis(
    df: pd.DataFrame,
    metric_col: str = "distance_perc",
    treatment_variant_name: str = 'treatment',
    control_variant_name: str = 'control',
    sign_convention: str = "lower_is_better",
    lift_type: str = "absolute",
    alpha: float = 0.05,
    min_valid_days: int = 5,
    pre_experiment_dist: Optional[pd.Series] = None,
    use_spend_weights: bool = False,
    test_method: str = "t-test",
    n_permutations: int = 10_000,
    random_seed: int = 42,
    goal_type_source: str = "contractual",
    verbose: bool = True,
    agg_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Full post-experiment analysis of a paired per-campaign A/B test: sign
    normalisation, aggregation, paired differences, CUPED, random-effects
    meta-analysis, overall test and BH-corrected segments. Negative Δ →
    treatment better. Methodology and details: experiment_utils.md, section 2.

    Parameters
    ----------
    df : one row per campaign × variant × date (required columns: see the .md)
    metric_col : 'distance_perc' (default), 'absolute_cpx' or a custom column
    treatment_variant_name, control_variant_name : `variant` values of each arm
    sign_convention : custom metrics only, 'lower_is_better' / 'higher_is_better'
    lift_type : 'absolute' (treatment − control) or 'relative' (÷ |control|)
    alpha : significance level
    min_valid_days : minimum paired days per campaign
    pre_experiment_dist : CUPED covariate per campaign_id (None = synthetic, testing only)
    use_spend_weights : weight CUPED and the overall test by campaign spend
    test_method : 't-test' (bootstrap z-test if spend-weighted) or 'permutation'
    n_permutations : permutations for test_method='permutation'
    random_seed : seed for the synthetic covariate, bootstrap and permutations
    goal_type_source : 'contractual' or 'operative' goal columns
    verbose : print the summary
    agg_df : REQUIRED for custom metrics (ignored, with a UserWarning, for
        'distance_perc' / 'absolute_cpx'). One row per campaign × variant with
        `campaign_id`, `variant`, `metric_value` (window-level value, in the metric's
        own direction; `sign_convention` is applied here), plus `spend_sum` if
        `use_spend_weights=True` and `campaign_type` / goal columns for the segments
        (a segment is skipped with a warning if its columns are missing). It replaces
        the daily-mean aggregation; `df` still gives Var(δ_c) and the min-days filter.

    Returns
    -------
    dict: 'overall', 'meta_analysis', 'campaign_level', 'segments',
    'cuped_stats', 'n_excluded'.
    """
    goal_type_col, goal_value_col = _validate_inputs(
        df, sign_convention, test_method, goal_type_source, lift_type,
    )

    df_clean = _preprocess(
        df, metric_col=metric_col, sign_convention=sign_convention,
        goal_value_col=goal_value_col,
    )
    n_raw    = df_clean["campaign_id"].nunique()

    # --- Aggregate to campaign × variant level ---
    if metric_col in ["distance_perc", "absolute_cpx"]:
        if agg_df is not None:
            warnings.warn(
                f"agg_df is ignored for metric_col='{metric_col}' (ratio-of-sums is "
                "computed from df).",
                UserWarning,
            )
        agg_metric = _aggregate_distance(
            df_clean, metric_=metric_col,
            goal_type_col=goal_type_col, goal_value_col=goal_value_col,
        )
    else:
        agg_metric = _prepare_agg_df(
            agg_df, df_clean, sign_convention,
            control_variant_name, treatment_variant_name, use_spend_weights,
            goal_type_col=goal_type_col, goal_value_col=goal_value_col,
        )

    daily_df   = _daily_pairs(df_clean, lift_type=lift_type, control_variant_name = control_variant_name, treatment_variant_name = treatment_variant_name)
    camp       = _build_campaign_table(
        agg_metric, daily_df, min_valid_days, lift_type=lift_type,
        control_variant_name=control_variant_name, treatment_variant_name=treatment_variant_name,
        goal_type_col=goal_type_col, goal_value_col=goal_value_col,
        flip_roas=metric_col in ("distance_perc", "absolute_cpx"),
    )
    n_excluded = n_raw - len(camp)

    if len(camp) < 3:
        raise ValueError(
            f"Only {len(camp)} campaign(s) passed quality filters "
            f"(min_valid_days={min_valid_days}). "
            "Lower the threshold or check the dataset."
        )

    # Resolve spend weights (None when equal-weighting)
    spend_weights = camp["total_spend_c"].values.astype(float) if use_spend_weights else None

    camp = _apply_cuped(camp, pre_experiment_dist, random_seed, weights=spend_weights)
    cuped_stats = {
        "theta":                  camp.attrs["cuped_theta"],
        "variance_reduction_pct": camp.attrs["cuped_variance_reduction"] * 100,
        "note": (
            "Synthetic pre-experiment covariate (replace for production)"
            if pre_experiment_dist is None
            else "Real pre-experiment data"
        ),
    }

    meta = _random_effects_meta(
        camp["delta_c"].values,
        camp["var_delta_c"].values,
    )

    if test_method == "permutation":
        overall = _permutation_test(
            camp["delta_c_cuped"].values,
            alpha,
            weights=spend_weights,
            n_perm=n_permutations,
            random_seed=random_seed,
        )
    elif use_spend_weights:
        overall = _weighted_hypothesis_test(
            camp["delta_c_cuped"].values,
            spend_weights,
            alpha,
            random_seed=random_seed,
        )
    else:
        overall = _hypothesis_test(camp["delta_c_cuped"].values, alpha)

    segment_cols = SEGMENT_COLS_BASE + [goal_type_col]
    segments = {
        col: _segment_analysis(
            camp, col, alpha,
            test_method=test_method,
            n_perm=n_permutations,
            random_seed=random_seed,
        )
        for col in segment_cols
        if col in camp.columns
    }

    if verbose:
        _print_summary(
            overall, meta, cuped_stats, segments,
            n_raw, n_excluded, alpha, use_spend_weights, test_method,
            metric_col=metric_col,
            sign_convention=sign_convention,
            lift_type=lift_type,
        )

    return {
        "overall":        overall,
        "meta_analysis":  meta,
        "campaign_level": camp,
        "segments":       segments,
        "cuped_stats":    cuped_stats,
        "n_excluded":     n_excluded,
    }


# ----------------------------------------------------------------------------------------------
# ------------------------------- Log-space ("log-t", BETA) functions --------------------------
# ----------------------------------------------------------------------------------------------
# Counterparts of the raw-space design and post-experiment functions above.
# They reuse the shared helpers (_prepare_agg_df, _random_effects_meta,
# _hypothesis_test, _weighted_hypothesis_test, _permutation_test, _segment_analysis).


def estimate_variance_scaling_logt_beta(
    daily_df,
    metric_calculator,
    campaign_col="campaign_id",
    date_col="date",
    candidate_days=None,
    min_windows_per_day_count=5,
    log_space=False,
    verbose=True,
):
    """
    BETA. `estimate_variance_scaling` with a `log_space` option: when True the
    fit runs on log(metric) (window values must be strictly positive). Returns
    the same dict plus 'log_space'. See experiment_utils.md, section 4.
    """
    c, a, r_squared, points, skipped, n_nonpositive_dropped = _variance_scaling_fit(
        daily_df, metric_calculator, campaign_col, date_col,
        candidate_days, min_windows_per_day_count, log_space,
    )

    if verbose:
        _print_variance_scaling(
            f"Variance-scaling fit ({'log' if log_space else 'raw'}-space): "
            f"variance(T) = c * T^-a",
            points, c, a, r_squared, skipped,
            n_nonpositive_dropped if log_space else 0,
        )

    return {
        "c": c, "a": a, "r_squared": r_squared, "points": points,
        "skipped_days": skipped, "log_space": log_space,
    }



def mde_by_campaigns_and_days_logt_beta(
    daily_df,
    metric_calculator,
    value_a,
    n_campaigns=None,
    n_days=None,
    campaigns_options=None,
    days_options=np.arange(7, 28),
    candidate_scaling_days=None,
    historical_rho=0.0,
    alpha=0.05,
    power=0.8,
    campaign_col="campaign_id",
    date_col="date",
    log_space=False,
    plot=True,
):
    """
    BETA. `mde_by_campaigns_and_days` with a `log_space` option: when True the
    variance fit runs on log(metric) and the MDE is the multiplicative effect
    (exp(d · σ) − 1) · 100 (`value_a` ignored). Same parameters and outputs
    otherwise. See experiment_utils.md, section 4.
    """
    scaling = estimate_variance_scaling_logt_beta(
        daily_df,
        metric_calculator,
        campaign_col=campaign_col,
        date_col=date_col,
        candidate_days=candidate_scaling_days,
        log_space=log_space,
        verbose=True,
    )
    c, a = scaling["c"], scaling["a"]

    # Calculate the std dev of our metric (in log-units when log_space=True)
    def sigma_delta(T):
        return np.sqrt(c * T ** (-a))

    # Reduce the std dev with CUPED
    def sigma_delta_cuped(T):
        return sigma_delta(T) * np.sqrt(max(0.0, 1 - historical_rho ** 2))

    # d (Cohen's d) * sigma = smallest detectable difference, in the space of the fit
    if log_space:
        # d * sigma is a log-ratio: exp() - 1 gives the relative MDE (no value_a needed)
        def mde_relative_pct(N, T):
            d = TTestPower().solve_power(effect_size=None, nobs=N, alpha=alpha, power=power, alternative="two-sided")
            return (np.exp(d * sigma_delta_cuped(T)) - 1) * 100
    else:
        # Dividing by value_a (the baseline mean) and multiplying by 100 converts
        # the absolute MDE (in the metric's own units) into a relative % MDE
        def mde_relative_pct(N, T):
            d = TTestPower().solve_power(effect_size=None, nobs=N, alpha=alpha, power=power, alternative="two-sided")
            return d * sigma_delta_cuped(T) / value_a * 100

    return _mde_by_mode(
        mde_relative_pct, daily_df, campaign_col, n_campaigns, n_days,
        campaigns_options, days_options, plot,
    )


# ---------------------------------------------------------------------------
# 1b. Preprocessing (log-t)
# ---------------------------------------------------------------------------

def _preprocess_logt_beta(
    df: pd.DataFrame,
    metric_col: str = "distance_perc",
    sign_convention: str = "lower_is_better",
    goal_value_col: str = "contractual_goal_value",
) -> pd.DataFrame:
    """
    BETA. Log-t `_preprocess`: distance_perc → 1 + distance_perc for every row
    (= cpx/goal, or actual/goal for ROAS; positive iff the metric is > 0),
    absolute_cpx kept raw; the ROAS sign is applied once, in
    `_build_campaign_table_logt_beta`. Custom 'higher_is_better' → 1/x (undone in
    `run_experiment_analysis_logt_beta`).
    """
    if metric_col not in df.columns:
        raise ValueError(
            f"metric_col='{metric_col}' not found in the DataFrame. "
            f"Available columns: {sorted(df.columns.tolist())}"
        )

    df = df.copy()

    _std_numerics = [
        "bids", "impressions", "spend", "clicks", "installs", "goal_event",
        "goal_event_value", goal_value_col, "distance_perc", "absolute_cpx"
    ]
    for col in _std_numerics:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if metric_col not in _std_numerics:
        df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], utc=True)

    if metric_col == "distance_perc":
        # 1 + distance_perc = cpx/goal (actual/goal for ROAS): a positive ratio to
        # goal for every goal type, so the log difference is a log-ratio and
        # campaigns beating their goal (distance_perc < 0) are kept. Zero-revenue
        # ROAS days give 0 and fail the positivity filter. No inversion: the ROAS
        # sign is applied once, to delta_c_log (variance is sign-invariant).
        df["metric_analysis"] = 1 + df["distance_perc"]
    elif metric_col == "absolute_cpx":
        # Raw CPx / ROAS; zero-revenue ROAS days fail the positivity filter.
        df["metric_analysis"] = df["absolute_cpx"]
    else:
        if sign_convention == "higher_is_better":
            df["metric_analysis"] = 1 / df[metric_col].astype(float)
        else:
            df["metric_analysis"] = df[metric_col].astype(float)

    return df


# ---------------------------------------------------------------------------
# 2c. Aggregate CPx -> % distance per campaign x variant (log-t)
# ---------------------------------------------------------------------------

def _aggregate_distance_logt_beta(
    df: pd.DataFrame,
    metric_: str = 'distance_perc',
    goal_type_col: str = "contractual_goal_type",
    goal_value_col: str = "contractual_goal_value",
) -> pd.DataFrame:
    """
    BETA. Log-t `_aggregate_distance`: distance_perc → 1 + dist_agg (= cpx/goal) for
    every goal type, as in `_preprocess_logt_beta`; no positivity filter here.
    """
    goal_types = df[goal_type_col].unique()
    parts = []

    for goal_type in goal_types:
        subset = df[df[goal_type_col] == goal_type].copy()
        num_col, den_col = _CPX_COLS.get(goal_type, ("spend", "goal_event"))

        agg = (
            subset.groupby(
                ["campaign_id", "variant", "campaign_type",
                 goal_type_col, goal_value_col],
                as_index=False,
            )
            .agg(
                num_sum=(num_col,  "sum"),
                den_sum=(den_col,  "sum"),
                spend_sum=("spend", "sum"),
            )
        )

        agg = agg[agg["den_sum"] > 0].copy()

        agg["cpx_agg"] = agg["num_sum"] / agg["den_sum"]
        if metric_ == 'distance_perc':
            agg["dist_agg"] = agg["cpx_agg"] / agg[goal_value_col] - 1
        elif metric_ == 'absolute_cpx':
            agg["dist_agg"] = agg["cpx_agg"]

        if metric_ == 'distance_perc':
            agg["metric_value"] = 1 + agg["dist_agg"]
        else:
            # absolute_cpx: kept as-is, NOT positivity-filtered here
            # (experiment_utils.md, section 5).
            agg["metric_value"] = agg["dist_agg"]

        parts.append(agg)

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# 3b. Daily paired log-ratios (log-t variance estimation)
# ---------------------------------------------------------------------------

def _daily_pairs_logt_beta(
    df: pd.DataFrame,
    treatment_variant_name: str = "treatment",
    control_variant_name: str = "base",
) -> pd.DataFrame:
    """
    BETA. Daily log-ratios log(treatment) − log(control); days where a variant is
    null or non-positive are excluded (with a warning).
    """
    variants = df[df["variant"].isin([control_variant_name, treatment_variant_name])].copy()

    pivot = variants.pivot_table(
        index=["campaign_id", "date"],
        columns="variant",
        values="metric_analysis",
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None

    pivot = pivot.dropna(subset=[control_variant_name, treatment_variant_name])

    valid = (pivot[control_variant_name] > 0) & (pivot[treatment_variant_name] > 0)
    dropped = int((~valid).sum())
    pivot = pivot[valid].copy()

    if dropped > 0:
        warnings.warn(
            f"{dropped} campaign-day(s) excluded: log-t requires strictly "
            "positive metric_analysis for both variants (pass a calculator "
            "returning e.g. `1 + distance_perc` if your metric can be "
            "non-positive).",
            UserWarning,
        )

    pivot["daily_delta_log"] = (
        np.log(pivot[treatment_variant_name]) - np.log(pivot[control_variant_name])
    )

    return pivot[["campaign_id", "date", "daily_delta_log"]]


# ---------------------------------------------------------------------------
# 4b. Campaign-level pairing and filtering (log-t)
# ---------------------------------------------------------------------------

def _build_campaign_table_logt_beta(
    agg_metric: pd.DataFrame,
    daily_pair_df: pd.DataFrame,
    min_valid_days: int,
    treatment_variant_name: str = "treatment",
    control_variant_name: str = "base",
    goal_type_col: str = "contractual_goal_type",
    goal_value_col: str = "contractual_goal_value",
    metric_col: str = "distance_perc",
) -> pd.DataFrame:
    """
    BETA. Log-t campaign table: delta_c_log and its variance; campaigns with a
    non-positive window metric are dropped (warns). For ROAS rows (distance_perc /
    absolute_cpx) delta_c_log is negated (higher ROAS = better → negative Δ); the
    single ROAS flip. metric_control / metric_treatment are shown as in the raw
    path (distance_perc / absolute_cpx, ROAS as −x).
    """
    _possible_idx = [
        "campaign_id", "campaign_type", goal_type_col, goal_value_col,
    ]
    index_cols = [c for c in _possible_idx if c in agg_metric.columns]

    wide = agg_metric.pivot_table(
        index=index_cols,
        columns="variant",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    wide = wide.dropna(subset=[control_variant_name, treatment_variant_name])

    valid = (wide[control_variant_name] > 0) & (wide[treatment_variant_name] > 0)
    dropped = int((~valid).sum())
    wide = wide[valid].copy()
    if dropped > 0:
        warnings.warn(
            f"{dropped} campaign(s) excluded: log-t requires a strictly "
            "positive window-level metric for both variants.",
            UserWarning,
        )

    wide["delta_c_log"] = (
        np.log(wide[treatment_variant_name]) - np.log(wide[control_variant_name])
    )

    is_roas = (
        wide[goal_type_col].isin(ROAS_GOAL_TYPES)
        if metric_col in ("distance_perc", "absolute_cpx") and goal_type_col in wide.columns
        else pd.Series(False, index=wide.index)
    )
    if is_roas.any():
        # Higher ROAS = better: negate delta_c_log (variance is sign-invariant).
        wide.loc[is_roas, "delta_c_log"] *= -1
    if metric_col == "distance_perc":
        # metric_value = 1 + dist_agg  =>  display dist_agg (ROAS as -dist_agg)
        for v in (control_variant_name, treatment_variant_name):
            wide[v] = wide[v] - 1
    if is_roas.any():
        for v in (control_variant_name, treatment_variant_name):
            wide.loc[is_roas, v] = -wide.loc[is_roas, v]

    wide = wide.rename(
        columns={
            control_variant_name:   "metric_control",
            treatment_variant_name: "metric_treatment",
        }
    )

    wide = _attach_total_spend(wide, agg_metric, treatment_variant_name, control_variant_name)

    return _attach_variance_and_filter(
        wide, daily_pair_df, "daily_delta_log", "var_delta_c_log", min_valid_days,
    )


# ---------------------------------------------------------------------------
# 5b. CUPED (log-t)
# ---------------------------------------------------------------------------

def _apply_cuped_logt_beta(
    camp: pd.DataFrame,
    pre_experiment_dist: Optional[pd.Series],
    random_seed: int,
    weights: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    BETA. CUPED on delta_c_log with X = log(pre-experiment metric, raw units, > 0);
    the synthetic / fallback covariate is built from delta_c_log.
    """
    df  = camp.copy()
    rng = np.random.default_rng(random_seed)

    # NOTE: metric_control/metric_treatment are re-expressed as -x for ROAS
    # (see `_build_campaign_table_logt_beta`) and may be non-positive, so the
    # synthetic/fallback covariate is built from delta_c_log, not their logs.
    if not isinstance(pre_experiment_dist, pd.Series):
        warnings.warn(
            "No pre-experiment data provided — using a synthetic log-space "
            "CUPED covariate. Pass a real pd.Series (indexed by campaign_id, "
            "in raw units — it is log-transformed internally) via "
            "`pre_experiment_dist` before using results in production.",
            UserWarning,
        )
        baseline_log = -df["delta_c_log"] / 2
        noise_sd     = baseline_log.std(ddof=1) * 0.8 if baseline_log.std(ddof=1) > 0 else 0.1
        synthetic    = baseline_log + rng.normal(0, noise_sd, size=len(df))
        df["pre_exp_dist_log"] = synthetic.values
    else:
        raw = df["campaign_id"].map(pre_experiment_dist)
        non_positive = (raw <= 0).sum()
        if non_positive > 0:
            raise ValueError(
                f"{non_positive} campaign(s) have a non-positive "
                "pre_experiment_dist value; log-t requires a strictly "
                "positive pre-experiment covariate (pass raw units — "
                "they are log-transformed internally)."
            )
        df["pre_exp_dist_log"] = np.log(raw)
        missing = df["pre_exp_dist_log"].isna().sum()
        if missing > 0:
            warnings.warn(
                f"{missing} campaign(s) have no pre-experiment covariate; "
                "falling back to in-experiment log-space average.",
                UserWarning,
            )
            fallback = -df["delta_c_log"] / 2
            df["pre_exp_dist_log"] = df["pre_exp_dist_log"].fillna(fallback)

    X = df["pre_exp_dist_log"].values.astype(float)
    Y = df["delta_c_log"].values.astype(float)

    df["delta_c_log_cuped"], theta, reduction = _cuped_adjust(X, Y, weights)

    df.attrs["cuped_theta"]              = theta
    df.attrs["cuped_variance_reduction"] = reduction

    return df


# ---------------------------------------------------------------------------
# Summary printer (log-t)
# ---------------------------------------------------------------------------

def _print_summary_logt_beta(
    overall: dict,
    meta: dict,
    cuped_stats: dict,
    segments: dict,
    n_raw: int,
    n_excluded: int,
    alpha: float,
    use_spend_weights: bool,
    test_method: str = "t-test",
    metric_col: str = "distance_perc",
    sign_convention: str = "lower_is_better",
) -> None:
    """BETA. Log-space counterpart of `_print_summary`."""
    sep = "=" * 64

    print(sep)
    print("  POST-EXPERIMENT ANALYSIS — JAMPP A/B TEST (LOG-T, BETA)")
    print(sep)

    weight_mode = "spend-weighted" if use_spend_weights else "equal-weighted"
    print(f"\nCAMPAIGNS")
    print(f"  In dataset          : {n_raw}")
    print(f"  Included in analysis: {overall['n_campaigns']}")
    print(f"  Excluded (low data) : {n_excluded}")
    print(f"  Metric              : {metric_col}  ({sign_convention}, log-ratio Δ)")
    print(f"  Weighting mode      : {weight_mode}")
    print(f"  Hypothesis test     : {test_method}")

    print(f"\nCUPED VARIANCE REDUCTION (log space)")
    print(f"  θ (covariate weight): {cuped_stats['theta']:+.4f}")
    print(f"  Variance reduced by : {cuped_stats['variance_reduction_pct']:.1f}%")
    print(f"  Covariate source    : {cuped_stats['note']}")

    sig_label = "✓  SIGNIFICANT" if overall["significant"] else "✗  NOT SIGNIFICANT"
    dir_label = overall["direction"].replace("_", " ").upper()
    mult_mean = (np.exp(overall["mean_delta"]) - 1) * 100
    mult_lo   = (np.exp(overall["ci_lower"]) - 1) * 100
    mult_hi   = (np.exp(overall["ci_upper"]) - 1) * 100

    test_label, stat_line = _test_label_and_stat_line(overall, alpha, use_spend_weights, test_method)

    print(f"\n{test_label}")
    print(f"  {sig_label}  |  {dir_label}")
    print(f"  Mean Δ (log, CUPED) : {overall['mean_delta']:+.4f}")
    print(f"  95% CI (log)        : [{overall['ci_lower']:+.4f},  {overall['ci_upper']:+.4f}]")
    print(f"  Mean Δ (multiplic.) : {mult_mean:+.2f}%")
    print(f"  95% CI (multiplic.) : [{mult_lo:+.2f}%,  {mult_hi:+.2f}%]")
    if stat_line:
        print(stat_line)
    print(f"  p-value             : {overall['p_value']:.4f}")
    print(f"  N campaigns         : {overall['n_campaigns']}")

    print(f"\nRANDOM-EFFECTS META-ANALYSIS  (DerSimonian-Laird, inverse-variance weighted, log space)")
    print(f"  Pooled Δ (log)      : {meta['pooled_estimate']:+.4f}  (SE = {meta['se']:.4f})")
    print(f"  95% CI (log)        : [{meta['ci_lower']:+.4f},  {meta['ci_upper']:+.4f}]")
    print(f"  τ² (heterogeneity)  : {meta['tau2']:.4f}")
    print(f"  I²                  : {meta['I2']:.1f}%")
    print(f"  Q statistic         : {meta['Q']:.2f}  (p = {meta['Q_pvalue']:.3f})")

    for col, seg_df in segments.items():
        if seg_df.empty:
            continue
        print(f"\nSEGMENT BREAKDOWN — {col.upper()}  (equal-weighted, BH FDR @ α={alpha}, log space)")
        hdr = f"  {'Segment':<28} {'N':>4}  {'Mean Δ (log)':>13}  {'p-value':>8}  {'BH-sig':>7}  Direction"
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for _, row in seg_df.iterrows():
            sig  = "Yes *" if row["significant_bh"] else "No"
            dirn = row["direction"].replace("_", " ")
            print(
                f"  {str(row['segment']):<28} {int(row['n_campaigns']):>4}"
                f"  {row['mean_delta']:>+13.4f}  {row['p_value_raw']:>8.4f}"
                f"  {sig:>7}  {dirn}"
            )

    print(f"\n{sep}")
    print("  SIGN CONVENTION")
    if metric_col not in ("distance_perc", "absolute_cpx") and sign_convention == "higher_is_better":
        print("  Positive Δ → treatment performs BETTER than control")
        print("  Negative Δ → treatment performs WORSE  than control")
        print(f"  ({metric_col} is higher-is-better: internally inverted (1/x) to")
        print("   stay positive for log-t, then un-inverted/sign-corrected in the")
        print("   returned values below, so Δ, metric_control/metric_treatment,")
        print("   and 'direction' all read in RAW units/direction — same")
        print("   sign_convention the caller passed in.)")
    else:
        print("  Negative Δ → treatment performs BETTER than control")
        print("  Positive Δ → treatment performs WORSE  than control")
        if metric_col in ("distance_perc", "absolute_cpx"):
            print("  (Cost metrics: lower = better.")
            print("   ROAS: log-ratio sign flipped so that higher ROAS = negative Δ;")
            print("   window values shown as -distance_perc/-absolute_cpx, identical to")
            print("   what `run_experiment_analysis` displays for the same ROAS campaign.)")
        else:
            print(f"  ({metric_col} is lower-is-better; no transform applied.)")
    print("  Δ is a log-ratio: log(treatment) − log(control).")
    print("  Multiplicative Δ = (exp(Δ) − 1) × 100 is the % change relative to control.")
    _print_caveats(use_spend_weights, alpha)
    print(f"  ⚠  BETA: log-t analysis path, not yet validated as thoroughly")
    print(f"     as the raw-space `run_experiment_analysis`.")
    print(sep)


# ---------------------------------------------------------------------------
# Public API (log-t)
# ---------------------------------------------------------------------------

def run_experiment_analysis_logt_beta(
    df: pd.DataFrame,
    metric_col: str = "distance_perc",
    treatment_variant_name: str = 'treatment',
    control_variant_name: str = 'control',
    sign_convention: str = "lower_is_better",
    alpha: float = 0.05,
    min_valid_days: int = 5,
    pre_experiment_dist: Optional[pd.Series] = None,
    use_spend_weights: bool = False,
    test_method: str = "t-test",
    n_permutations: int = 10_000,
    random_seed: int = 42,
    goal_type_source: str = "contractual",
    verbose: bool = True,
    agg_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    BETA. Log-space `run_experiment_analysis` for experiments designed with
    `mde_by_campaigns_and_days_logt_beta(log_space=True)`: the paired difference is
    log(treatment) − log(control), and effects are also reported as % change,
    (exp(Δ) − 1) × 100. Same parameters minus `lift_type` (incl. `agg_df`, required
    for custom metrics); metrics must be strictly positive; `pre_experiment_dist` is
    in raw units.

    Sign: negative Δ → treatment better, EXCEPT custom 'higher_is_better' metrics,
    whose Δ is returned in raw direction (positive = better). The `direction`
    labels are always right. Details: experiment_utils.md, sections 3 and 4.

    Returns
    -------
    As `run_experiment_analysis`, with log-space columns and the
    '*_multiplicative_pct' keys in 'overall'.
    """
    goal_type_col, goal_value_col = _validate_inputs(
        df, sign_convention, test_method, goal_type_source,
    )

    df_clean = _preprocess_logt_beta(
        df, metric_col=metric_col, sign_convention=sign_convention,
        goal_value_col=goal_value_col,
    )
    n_raw    = df_clean["campaign_id"].nunique()

    if metric_col in ["distance_perc", "absolute_cpx"]:
        if agg_df is not None:
            warnings.warn(
                f"agg_df is ignored for metric_col='{metric_col}' (ratio-of-sums is "
                "computed from df).",
                UserWarning,
            )
        agg_metric = _aggregate_distance_logt_beta(
            df_clean, metric_=metric_col,
            goal_type_col=goal_type_col, goal_value_col=goal_value_col,
        )
    else:
        agg_metric = _prepare_agg_df(
            agg_df, df_clean, sign_convention,
            control_variant_name, treatment_variant_name, use_spend_weights,
            goal_type_col=goal_type_col, goal_value_col=goal_value_col,
            log_space=True,
        )

    daily_df = _daily_pairs_logt_beta(
        df_clean,
        control_variant_name=control_variant_name,
        treatment_variant_name=treatment_variant_name,
    )
    camp = _build_campaign_table_logt_beta(
        agg_metric, daily_df, min_valid_days,
        control_variant_name=control_variant_name, treatment_variant_name=treatment_variant_name,
        goal_type_col=goal_type_col, goal_value_col=goal_value_col,
        metric_col=metric_col,
    )
    n_excluded = n_raw - len(camp)

    if len(camp) < 3:
        raise ValueError(
            f"Only {len(camp)} campaign(s) passed quality filters "
            f"(min_valid_days={min_valid_days}, strictly-positive-metric requirement). "
            "Lower the threshold, check the dataset, or shift the metric positive."
        )

    if metric_col not in ("distance_perc", "absolute_cpx") and sign_convention == "higher_is_better":
        # Undo _preprocess_logt_beta's 1/x (same for every campaign) so the output is in
        # raw direction: positive Δ = better. Opposite Δ sign from run_experiment_analysis
        # (see experiment_utils.md, section 3).
        camp["delta_c_log"]      = -camp["delta_c_log"]
        camp["metric_control"]   = 1 / camp["metric_control"]
        camp["metric_treatment"] = 1 / camp["metric_treatment"]

    spend_weights = camp["total_spend_c"].values.astype(float) if use_spend_weights else None

    camp = _apply_cuped_logt_beta(camp, pre_experiment_dist, random_seed, weights=spend_weights)
    cuped_stats = {
        "theta":                  camp.attrs["cuped_theta"],
        "variance_reduction_pct": camp.attrs["cuped_variance_reduction"] * 100,
        "note": (
            "Synthetic pre-experiment covariate (replace for production)"
            if pre_experiment_dist is None
            else "Real pre-experiment data (raw units, log-transformed internally)"
        ),
    }

    meta = _random_effects_meta(
        camp["delta_c_log"].values,
        camp["var_delta_c_log"].values,
    )

    if test_method == "permutation":
        overall = _permutation_test(
            camp["delta_c_log_cuped"].values,
            alpha,
            weights=spend_weights,
            n_perm=n_permutations,
            random_seed=random_seed,
        )
    elif use_spend_weights:
        overall = _weighted_hypothesis_test(
            camp["delta_c_log_cuped"].values,
            spend_weights,
            alpha,
            random_seed=random_seed,
        )
    else:
        overall = _hypothesis_test(camp["delta_c_log_cuped"].values, alpha)

    overall["mean_delta_multiplicative_pct"] = float((np.exp(overall["mean_delta"]) - 1) * 100)
    overall["ci_lower_multiplicative_pct"]   = float((np.exp(overall["ci_lower"]) - 1) * 100)
    overall["ci_upper_multiplicative_pct"]   = float((np.exp(overall["ci_upper"]) - 1) * 100)

    segment_cols = SEGMENT_COLS_BASE + [goal_type_col]
    segments = {}
    for col in segment_cols:
        if col not in camp.columns:
            continue
        seg_df = _segment_analysis(
            camp.rename(columns={"delta_c_log_cuped": "delta_c_cuped"}),
            col, alpha,
            test_method=test_method,
            n_perm=n_permutations,
            random_seed=random_seed,
        )
        segments[col] = seg_df

    if metric_col not in ("distance_perc", "absolute_cpx") and sign_convention == "higher_is_better":
        # Δ is now in raw direction, but the shared tests label "negative = treatment_better":
        # flip the `direction` labels to match.
        _FLIP_DIRECTION = {
            "treatment_better": "control_better",
            "control_better":   "treatment_better",
        }
        overall["direction"] = _FLIP_DIRECTION[overall["direction"]]
        for seg_df in segments.values():
            if not seg_df.empty:
                seg_df["direction"] = seg_df["direction"].map(_FLIP_DIRECTION)

    if verbose:
        _print_summary_logt_beta(
            overall, meta, cuped_stats, segments,
            n_raw, n_excluded, alpha, use_spend_weights, test_method,
            metric_col=metric_col,
            sign_convention=sign_convention,
        )

    return {
        "overall":        overall,
        "meta_analysis":  meta,
        "campaign_level": camp,
        "segments":       segments,
        "cuped_stats":    cuped_stats,
        "n_excluded":     n_excluded,
    }
