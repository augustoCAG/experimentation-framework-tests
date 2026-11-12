# Experiment utilities — `experiment_utils_functions.py`

Utilities to design and analyse Jampp campaign-level A/B experiments with a **paired, per-campaign design**: every campaign runs both variants, and each campaign contributes one treatment − control difference.

This document holds the full explanations. The docstrings in the `.py` file are kept short and point here.

```python
import experiment_utils_functions as eu

results = eu.run_experiment_analysis(df)                 # distance_perc (default)
results = eu.run_experiment_analysis(                     # custom metric, e.g. CPx in USD
    df, metric_col="cpi_usd", sign_convention="lower_is_better",
    lift_type="relative",                                 # δ = (treat − ctrl) / |ctrl|
)
```

## Contents

| Area | Public functions |
|---|---|
| Design (pre-experiment) | `check_spend_balance`, `estimate_variance_scaling`, `mde_by_campaigns_and_days` |
| Post-experiment analysis, raw space | `run_experiment_analysis` |
| Log space ("log-t", **BETA**) | `estimate_variance_scaling_logt_beta`, `mde_by_campaigns_and_days_logt_beta`, `run_experiment_analysis_logt_beta` |

1. [Experiment design](#1-experiment-design)
2. [Post-experiment analysis (raw space)](#2-post-experiment-analysis-raw-space)
3. [Sign convention](#3-sign-convention)
4. [Log space ("log-t", BETA)](#4-log-space-log-t-beta)
5. [Pipeline internals](#5-pipeline-internals)
6. [Caveats and known issues](#6-caveats-and-known-issues)

---

## 1. Experiment design

### `check_spend_balance` — spend Sample Ratio Mismatch

Checks for spend imbalance between treatment and control (an SRM check adapted for spend). Two orthogonal checks run per campaign:

1. **Systematic imbalance**: a paired t-test on the daily `spend_treat − spend_ctrl` against H₀ = 0, combined with an aggregate imbalance threshold. Detects a consistent directional bias.
2. **Episodic imbalance**: counts days where the daily spend ratio satisfies `|spend_treat / (spend_treat + spend_ctrl) − 0.5| > spike_threshold`. Catches volatile allocation that averages out over the full window but invalidates individual days.

A campaign is also reported if it has fewer valid days than `min_days`. Only days where **both** variants have positive spend are used. The p-values of the t-tests are corrected with Benjamini-Hochberg FDR across campaigns.

| Parameter | Meaning |
|---|---|
| `df` | Columns `campaign_id`, `date`, `variant`, `spend` (same format as the input of `run_experiment_analysis`). |
| `treatment_variant_name`, `control_variant_name` | Values of `variant` that identify each arm (defaults `'treatment'` and `'base'`). |
| `threshold` | Aggregate imbalance threshold for the systematic flag (default 0.03 = 3%). Flagged only when the t-test is significant **and** `|agg_imbalance_pct| > threshold × 100`. |
| `spike_threshold` | Per-day threshold for the episodic flag (default 0.10). |
| `min_days` | Fewer valid paired days → flagged as low-data (default 5). |
| `alpha` | Significance level of the t-test with BH FDR correction (default 0.05). |
| `verbose` | Print a diagnostic summary (default True). |

Returns `{'campaign_level': DataFrame, 'summary': dict}`:

- `campaign_level`, one row per campaign: `campaign_id, n_valid_days, spend_treatment, spend_control, agg_ratio, agg_imbalance_pct, t_stat, p_value, n_spike_days, spike_dates, flag_low_days, t_significant, flag_systematic, flag_episodic, flagged`. `t_significant` is the BH-corrected result; `p_value` is the raw p-value (no BH-adjusted p-value column is produced).
- `summary`: `global_spend_treatment, global_spend_control, global_ratio, global_imbalance_pct, n_campaigns_total, n_campaigns_flagged, n_flag_systematic, n_flag_episodic, n_flag_low_days, pct_within_1pct, pct_within_3pct, pct_within_5pct`.

### `estimate_variance_scaling` — how noise shrinks with more days

Empirically estimates how the variance of a campaign-level metric changes with the aggregation window length T, instead of assuming a fixed 1/T law.

For each campaign and each window length T, every contiguous T-day slice of its daily history is aggregated (raw columns summed, like the ratio-of-sums in `run_experiment_analysis`) and passed through `metric_calculator`. The variance of those values approximates how noisy a T-day estimate is for that campaign. Averaging across campaigns gives one point per T, and the power law

    variance(T) = c · T^-a

is fit by OLS on log(T) vs log(variance). So both the scale `c` and the rate `a` come from the data. `a = 1` is the naive "variance of a mean" case; autocorrelation or seasonality in the daily data can make `a < 1`. The verbose report flags `|a − 1| > 0.3`.

It is called automatically by `mde_by_campaigns_and_days`, and is public so the fit can be inspected on its own.

| Parameter | Meaning |
|---|---|
| `daily_df` | Raw daily per-campaign data (one row per campaign per day) with the columns `metric_calculator` needs: spend, installs, goal_event, clicks, impressions, goal_event_value, and the goal type / value columns the calculator reads (see its `goal_type` parameter). |
| `metric_calculator` | Takes a dict/row of **summed** raw columns over a window and returns the metric, e.g. `absolute_cpx_calculator` / `distance_perc_calculator` from `utils_functions.py`. Non-numeric columns (e.g. the goal type) are passed with the window's first value. |
| `campaign_col`, `date_col` | Column names of the campaign and the date. |
| `candidate_days` | Window lengths T to test. Default: up to 6 values from 3 days to half of the **longest** campaign history. |
| `min_windows_per_day_count` | Minimum number of T-day windows a campaign needs to contribute at a given T (default 5). T values where no campaign qualifies are skipped and reported. At least 2 valid T values are needed, otherwise a `ValueError` is raised. |
| `verbose` | Print the empirical points, the fitted curve and any skipped T. |

Returns `{'c', 'a', 'r_squared', 'points', 'skipped_days'}`, where `points` is a DataFrame of T vs the empirical and fitted variance.

### `mde_by_campaigns_and_days` — minimum detectable effect

MDE for the paired, per-campaign design. It encodes these assumptions:

1. **Paired per-campaign design**: each campaign contributes one treatment − control difference, so the number of campaigns N is the sample size of a one-sample (paired) two-sided t-test (`statsmodels` `TTestPower`; defaults `alpha=0.05`, `power=0.8`).
2. **Noise**: the variance of a T-day estimate follows `variance(T) = c · T^-a`, with `c` and `a` fitted on the historical daily data by `estimate_variance_scaling` instead of assuming `a = 1`. `sqrt(variance(T))` is used as the std dev σ(T) of the per-campaign difference.
3. **CUPED (optional)**: σ(T) is scaled by `sqrt(1 − historical_rho²)`. The default `historical_rho = 0` assumes no CUPED benefit (team decision).
4. **Relative MDE (%)** = `d · σ(T) / value_a · 100`, where `d` is the standardised effect size (Cohen's d) solved by `TTestPower` and `value_a` is the baseline mean of the metric. In log space it is `(exp(d · σ(T)) − 1) · 100` instead (see [section 4](#4-log-space-log-t-beta)).

What it returns depends on which of `n_campaigns` / `n_days` are fixed:

| Fixed | Output |
|---|---|
| neither | full heatmap and DataFrame over `days_options` × `campaigns_options` (same shape as the legacy `mde_matrix` in `legacy files/experiments_utils_legacy.py`) |
| `n_days` | MDE across `campaigns_options` (line plot + DataFrame) |
| `n_campaigns` | MDE across `days_options` (line plot + DataFrame) |
| both | a single MDE value (float), also printed |

| Parameter | Meaning |
|---|---|
| `daily_df`, `metric_calculator` | Passed straight to `estimate_variance_scaling` (its fit is printed). |
| `value_a` | Baseline mean of the metric across campaigns (e.g. `ref_df[key_metric].mean()`), only used to turn the absolute MDE into a relative % (same convention as the legacy `mde_matrix`). |
| `campaigns_options` | Campaign counts shown when `n_campaigns` is free. Default: 3 values from 5 up to the number of distinct campaigns in `daily_df`. |
| `days_options` | Day counts shown when `n_days` is free. Default `np.arange(7, 28)` (same as the legacy `mde_matrix`). |
| `candidate_scaling_days` | Forwarded as `candidate_days`: the window lengths used to **fit** the curve, independent of `days_options`. |
| `historical_rho` | Correlation between a campaign's pre-experiment covariate and its treatment effect, for the optional CUPED reduction. |
| `alpha`, `power` | Significance level and target power of the one-sample t-test. |
| `plot` | Whether to draw the heatmap / line plot. |

---

## 2. Post-experiment analysis (raw space)

### Methodology of `run_experiment_analysis`

1. **Sign normalisation**: a `metric_analysis` column with the uniform [sign convention](#3-sign-convention). Goal type / value come from the columns selected by `goal_type_source`.
2. **Aggregation**: for `distance_perc` / `absolute_cpx`, ratio-of-sums over the full experiment window (avoids the bias of averaging daily ratios when daily spend is non-uniform). For custom metrics, the window-level value per campaign × variant is supplied by the caller in `agg_df` (e.g. a ratio-of-sums); the daily `df` still provides Var(δ_c) and the min-days / variance filter.
3. **Paired differences**: `δ_c = metric_treatment_c − metric_control_c`, or `(treatment − control) / |control|` with `lift_type='relative'`. Var(δ_c) = Var(daily paired differences) / n_days.
4. **Campaign filters**: campaigns with fewer than `min_valid_days` paired days, or an undefined / non-positive variance, are excluded (with a warning).
5. **CUPED**: δ_c is adjusted with a pre-experiment covariate, which lowers the variance without changing the expected value. A synthetic covariate is used if none is provided (testing only).
6. **Random-effects meta-analysis**: DerSimonian-Laird pooling across campaigns gives the pooled Δ, τ² and I². It uses inverse-variance weights on the non-CUPED δ_c.
7. **Hypothesis test**: one-sample t-test, spend-weighted bootstrap test, or sign-flip permutation test on the CUPED-adjusted δ_c (H₀: Δ = 0).
8. **Segment analysis**: per-segment tests (by `campaign_type` and by goal type) with Benjamini-Hochberg FDR correction. Always equal-weighted, and segments with fewer than 3 campaigns are skipped.

At least 3 campaigns must pass the filters, otherwise a `ValueError` is raised.

### Parameters of `run_experiment_analysis`

| Parameter | Meaning |
|---|---|
| `df` | One row per campaign × variant × date. Always needs `date`, `campaign_id`, `variant`, `<metric_col>`. For `distance_perc` / `absolute_cpx` it also needs `campaign_type`, `spend`, `installs`, `goal_event`, `clicks`, `goal_event_value`, and the goal columns selected by `goal_type_source`. For custom metrics only the columns you use are needed (the goal columns are still used for the goal-type segment breakdown). |
| `agg_df` (`None`) | **Required for custom metrics** (ignored with a `UserWarning` for `distance_perc` / `absolute_cpx`). One row per campaign × variant: `campaign_id`, `variant`, `metric_value` (window-level, in the metric's own direction; `sign_convention` is applied to it like to the daily values), `spend_sum` (only if `use_spend_weights=True`), and `campaign_type` / goal columns for the segments (a segment is skipped with a warning if its columns are missing). `ValueError` if `agg_df` or a required column is missing, a campaign × variant is duplicated, a variant name matches neither `control_variant_name` nor `treatment_variant_name`, or a campaign in `df` is absent from `agg_df`; campaigns only in `agg_df` are ignored with a warning. |
| `metric_col` (`'distance_perc'`) | Pre-computed numeric column. `'distance_perc'` and `'absolute_cpx'` use ratio-of-sums CPx aggregation with the per-row ROAS flip; any other column is a custom metric and requires `agg_df`. |
| `treatment_variant_name`, `control_variant_name` | Values of `variant` (defaults `'treatment'`, `'control'`). |
| `sign_convention` (`'lower_is_better'`) | Only for custom metrics (ignored for `distance_perc` / `absolute_cpx`). `'lower_is_better'`, e.g. CPx in USD: no change. `'higher_is_better'`, e.g. CTR or conversion rate: values multiplied by −1 internally. |
| `lift_type` (`'absolute'`) | `'absolute'`: δ_c = treatment − control, in the metric's units; best when all campaigns share the same scale (e.g. `distance_perc`). `'relative'`: δ_c = (treatment − control) / \|control\|, a dimensionless fraction; use when campaigns sit on very different scales (e.g. mixing CPI and CPA in USD) so each contributes equally. Days / campaigns where control is 0 are excluded. |
| `alpha` (0.05) | Significance level. |
| `min_valid_days` (5) | Minimum days of valid paired data per campaign. |
| `pre_experiment_dist` | Pre-experiment metric per `campaign_id` (pd.Series), in the same scale / lift space as δ_c. Campaigns missing from it fall back to their in-experiment average. If None, a synthetic covariate is generated (testing only, replace for production). |
| `use_spend_weights` (False) | True weights campaigns by total spend in CUPED and the overall test ("what was the effect across portfolio spend?") and needs `spend`. Segments are always equal-weighted. |
| `test_method` (`'t-test'`) | `'t-test'`: one-sample t-test (equal-weighted) or bootstrap z-test (spend-weighted). `'permutation'`: sign-flip permutation test, no distributional assumptions, only exchangeability of signs under H₀; same point estimate and CI, only the p-value changes. |
| `n_permutations` (10,000) | Permutations for the permutation test (50,000 gives finer p-values on small samples). |
| `random_seed` (42) | Seed for the synthetic covariate, permutation and bootstrap. |
| `goal_type_source` (`'contractual'`) | `'contractual'` uses `contractual_goal_type` / `contractual_goal_value` (the agreed goal). `'operative'` uses `operative_goal_type` / `operational_goal_value` (the goal actually applied, which can differ). Used throughout: ROAS flip, aggregation, campaign indexing and goal-type segments. |
| `verbose` (True) | Print the formatted summary. |

Returns a dict:

| Key | Content |
|---|---|
| `overall` | Overall test: `mean_delta, se, ci_lower, ci_upper, p_value, significant, direction, n_campaigns, alpha`, plus `t_stat`, `z_stat` + `n_bootstrap`, or `n_permutations` depending on the test. |
| `meta_analysis` | `pooled_estimate, se, ci_lower, ci_upper, tau2, I2, Q, Q_pvalue, k`. |
| `campaign_level` | Per-campaign table (`delta_c`, `var_delta_c`, `delta_c_cuped`, `total_spend_c`, …). |
| `segments` | BH-corrected segment results, a dict of DataFrames per segment column. |
| `cuped_stats` | `theta`, `variance_reduction_pct`, `note`. |
| `n_excluded` | Campaigns excluded by the quality filters. |

---

## 3. Sign convention

**Negative Δ → treatment performed BETTER than control. Positive Δ → WORSE.**

Every post-experiment function produces the same convention whatever the metric's own direction:

- **`distance_perc` / `absolute_cpx`**: cost metrics (CPI, CPA, CPC) are used as-is (lower = better). ROAS rows are flipped per row so that higher ROAS reads as negative Δ. Raw space negates the values (−x). Log space negates `delta_c_log` for ROAS rows (custom `higher_is_better` metrics are inverted instead, see section 4) and reports `metric_control` / `metric_treatment` of ROAS campaigns as −x, so they read exactly as in raw space.
- **Custom `metric_col`**: `sign_convention='lower_is_better'` (default) applies no change; `'higher_is_better'` flips the metric (raw: −x, log: 1/x).

**One exception (log space, custom `higher_is_better` metric).** `run_experiment_analysis_logt_beta` undoes its internal 1/x flip before returning. Its Δ is therefore in the metric's own direction (**positive = treatment better**), the opposite sign of `run_experiment_analysis` for the same input. The `direction` labels (`'treatment_better'` / `'control_better'`) and the printed summary are correct in both.

Worked example, for a higher-is-better metric with treatment = 12 and control = 10 (treatment is better):

| Function | Internally | Returned Δ | `direction` |
|---|---|---|---|
| `run_experiment_analysis` | −12 vs −10 → Δ = −2 | **−2** (negative = better) | `treatment_better` |
| `run_experiment_analysis_logt_beta` | 1/12 vs 1/10 → log ratio = −0.18 | **+0.18** = log(12/10) (positive = better) | `treatment_better` |

**When in doubt, read `direction`, not the sign of Δ.**

---

## 4. Log space ("log-t", BETA)

The `_logt_beta` functions are temporary and not yet validated as thoroughly as the raw-space ones.

### Why

Neither `absolute_cpx` nor `distance_perc` is normally distributed: both are right-skewed ratio metrics with most mass at small values and a long right tail. Every t-test used here (including `TTestPower` in the MDE) assumes the aggregated metric is approximately normal / t distributed, and that assumption fits **log(metric)** much better than the raw metric.

Use the log-t analysis when the experiment was **designed** with `mde_by_campaigns_and_days_logt_beta(log_space=True)`, i.e. the MDE was planned as a multiplicative effect. Analysing it with the raw-space function would test a different quantity (a raw mean difference) than the one the MDE was computed for (a log-ratio).

### Design: `estimate_variance_scaling_logt_beta`, `mde_by_campaigns_and_days_logt_beta`

Same method, parameters, defaults and outputs as the raw versions, plus `log_space` (default False). With `log_space=True`:

- every window value is log-transformed before its variance is computed, so the variance(T) fit runs on log(metric);
- Cohen's d from `TTestPower` is applied to the log-space σ, giving an effect size in log units;
- the relative MDE becomes the multiplicative effect `(exp(d · σ(T)) − 1) · 100`, and `value_a` is ignored.

Window values must be strictly positive. That holds for `absolute_cpx`; for `distance_perc`, pass a calculator that returns `1 + distance_perc` (i.e. cpx / goal). With `log_space=True`, non-positive values are dropped and counted in the verbose report, and a warning is printed when `daily_df` has a negative `distance_perc` column.

With `log_space=False` the numbers are the same as the raw functions. Only the printed header (`"(raw-space)"`) and the extra returned key `'log_space'` differ.

### Analysis: `run_experiment_analysis_logt_beta`

Parameters are the same as `run_experiment_analysis`, except there is no `lift_type`. The differences:

- **Paired difference**: `delta_c_log = log(metric_treatment) − log(metric_control)`. A difference of logs (structurally the 'absolute' form applied to log(metric)) is a proportional change, already scale-free like `lift_type='relative'`, hence no `lift_type`.
- **Strict positivity**: `metric_analysis` must be strictly positive for both variants, every day, for a campaign-day to count, and the window aggregate must be positive for a campaign to count. Excluded days / campaigns are reported with warnings. Shift metrics that can be zero or negative first (e.g. precompute `1 + distance_perc`).
- **Flipping "higher is better"**:
  - *Custom `higher_is_better` metrics* are flipped by **inversion** (1/x) instead of negation, because negating a positive rate would make it negative and log() would fail. The raw metric must be strictly positive (true for rate metrics like ITE / ITI).
  - *`distance_perc`* is analysed as `1 + distance_perc` (= cpx / goal, or actual / goal for ROAS) for every goal type. It is a positive ratio to the goal, so campaigns beating their goal (negative `distance_perc`) are kept and the log difference is a log-ratio. Because the goal cancels in the difference, the result equals the `absolute_cpx` analysis. *ROAS rows* are not inverted: they are positive iff actual ROAS > 0, so zero-revenue days fail the positivity filter. The sign is applied once, in `_build_campaign_table_logt_beta`: `delta_c_log` is negated for ROAS rows (the daily variance is sign-invariant, so it needs no flip).
- **Putting values back**: `metric_control` / `metric_treatment` are shown as in the raw path (`distance_perc` / `absolute_cpx`, ROAS as −x). For custom `higher_is_better` metrics the inversion is undone for every campaign before CUPED, and the `direction` labels are flipped accordingly (see the [exception](#3-sign-convention)).
- **`pre_experiment_dist`**: in **raw** units (e.g. pre-experiment `absolute_cpx`), log-transformed internally, and it must be strictly positive (a `ValueError` is raised otherwise). With `metric_col='distance_perc'` pass `1 + pre-experiment distance_perc` (= cpx / goal), the same scale as the analysed metric.
- **Reported effects**: on the log scale and as the multiplicative effect `(exp(Δ) − 1) × 100` (% change vs control). `overall` gains `mean_delta_multiplicative_pct`, `ci_lower_multiplicative_pct` and `ci_upper_multiplicative_pct`, which are directly comparable to the MDE % of `mde_by_campaigns_and_days_logt_beta(log_space=True)`. `campaign_level` uses the columns `delta_c_log`, `var_delta_c_log`, `delta_c_log_cuped`.

---

## 5. Pipeline internals

The raw pipeline is `_preprocess → _aggregate_distance | _prepare_agg_df → _daily_pairs → _build_campaign_table → _apply_cuped → _random_effects_meta + test → _segment_analysis → _print_summary`. The log-t pipeline mirrors it with the `_logt_beta` versions; the meta-analysis, tests and segment analysis are shared.

- **`_preprocess` / `_preprocess_logt_beta`**: coerce types and build `metric_analysis` with the sign convention. Raw: no sign change for `distance_perc` / `absolute_cpx` (the raw column; the ROAS flip happens in `_build_campaign_table`), × −1 for custom `higher_is_better`. Log: `distance_perc` → `1 + distance_perc` for every row, `absolute_cpx` raw (ROAS sign applied later, in `_build_campaign_table_logt_beta`), custom `higher_is_better` → 1/x.
- **`_build_campaign_table` / `_build_campaign_table_logt_beta`**: for `distance_perc` / `absolute_cpx`, the control / treatment window values of ROAS rows are multiplied by −1 **before** δ_c is computed. In the log version, `delta_c_log` is negated for ROAS rows instead (and the displayed window values re-expressed as −x). This is the only ROAS flip in each path; the daily variance (`_daily_pairs`) is sign-invariant, so it needs none. Custom metrics are not flipped here (they use `sign_convention`).
- **`_aggregate_distance`**: ratio-of-sums per campaign × variant, `CPx_agg = Σ numerator / Σ denominator`, `dist_agg = CPx_agg / goal_value − 1` (`distance_perc`) or `CPx_agg` (`absolute_cpx`); no sign change. Numerator / denominator per goal type: CPI spend / installs, CPA spend / goal_event, CPC spend / clicks, ROAS goal_event_value / spend. Rows whose denominator sums to zero are dropped.
- **`_aggregate_distance_logt_beta`**: same aggregation, but `distance_perc` becomes `1 + dist_agg` (= cpx / goal) for every goal type, as in `_preprocess_logt_beta`, so the window estimate and the daily values follow one convention. No positivity filter here. Positivity is enforced by `_daily_pairs_logt_beta` and `_build_campaign_table_logt_beta`.
- **`_prepare_agg_df`**: custom metrics; validates the caller's window-level `agg_df` and applies `sign_convention` (raw: −x, log: 1/x). Same output column `metric_value`, so the campaign table is metric-agnostic.
- **`_daily_pairs` / `_daily_pairs_logt_beta`**: daily paired differences (absolute, relative, or log-ratio) used for the variance estimate. Days where a variant is null (or non-positive in log space, or control = 0 with `lift_type='relative'`) are excluded.
- **`_build_campaign_table` / `_logt_beta`**: window-level point estimate δ_c plus variance `Var(daily delta) / n_valid_days`, total spend per campaign, and the min-days / variance filter (shared helpers `_attach_total_spend`, `_attach_variance_and_filter`).
- **`_apply_cuped` / `_logt_beta`**: `δ_cuped = δ − θ (X − X̄)`, θ = Cov(δ, X) / Var(X). With spend weights, θ, X̄ and the variances are weighted so CUPED targets the same estimand as the weighted test (shared core `_cuped_adjust`). In log space X = log(pre-experiment metric), and the synthetic / fallback covariate is built from `delta_c_log`, because `metric_control` / `metric_treatment` may be non-positive after the ROAS re-expression.
- **`_random_effects_meta`**: DerSimonian-Laird, giving the pooled estimate with a 95% CI, τ², I², Q and its p-value.
- **`_hypothesis_test`**: one-sample t-test on the CUPED-adjusted δ (H₀: E[δ] = 0).
- **`_weighted_hypothesis_test`**: estimand = spend-weighted mean Σ(w·δ) / Σw ("average effect across total portfolio spend" rather than "typical effect across campaigns"). The SE comes from a bootstrap that resamples campaigns with their weights (2,000 resamples), with a normal approximation, since the t-distribution is not appropriate for weighted means with arbitrary weights.
- **`_permutation_test`**: under H₀ each campaign's sign is exchangeable, so random sign flips give the null distribution. The p-value is floored at 1 / n_perm. The CI uses the t-test SE (equal weights) or the permutation SE with a normal critical value (spend weights).
- **`_segment_analysis`**: per-segment t-test or permutation test, BH FDR across the segments of a column (shared helper `_benjamini_hochberg`). Segments with fewer than 3 campaigns are skipped.

---

## 6. Caveats and known issues

- **Peeking**: the tests use a fixed-horizon design. If results were monitored during the experiment, the nominal α is anti-conservative; consider an mSPRT correction or Bonferroni α / n_peeks.
- **Synthetic CUPED covariate**: without `pre_experiment_dist` a synthetic covariate is generated (with a warning). Results are for testing only.
- **Known bug, not fixed**: a custom metric with **no `spend` column** raises `AttributeError: 'int' object has no attribute 'fillna'` (in the total-spend computation), even with `use_spend_weights=False`.
- **No BH-adjusted p-values in `check_spend_balance`**: older docs listed a `p_value_bh` column; it was never produced. Use `t_significant`.
- **Default control name**: `check_spend_balance` defaults to `control_variant_name='base'`, while the `run_experiment_analysis*` functions default to `'control'`.
- **Sign exception in log space**: see [section 3](#3-sign-convention).
