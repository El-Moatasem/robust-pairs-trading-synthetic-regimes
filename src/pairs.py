from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint


def estimate_cointegrating_regression(y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
    """Estimates an OLS cointegrating regression between two log-price asset series.

    Fits the log-linear model ln(y) = α + β · ln(x) + ε to
    determine the cointegration intercept α, hedge ratio β, and spread residuals.

    Args:
        y (pd.Series): Price series for the dependent asset.
        x (pd.Series): Price series for the independent asset.

    Returns:
        tuple[float, float, pd.Series]: A tuple containing:
            - **alpha** (float): Intercept term of the log-linear regression.
            - **beta** (float): Hedge ratio (slope coefficient) of the regression.
            - **residual** (pd.Series): Residual spread series computed as ln(y) - α - β·ln(x).

    Raises:
        ValueError: If fewer than 60 aligned observations exist after dropping missing values.
    """
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    if len(frame) < 60:
        raise ValueError("At least 60 aligned observations are required.")
    fit = OLS(np.log(frame["y"]), add_constant(np.log(frame["x"]))).fit()
    alpha = float(fit.params.iloc[0])
    beta = float(fit.params.iloc[1])
    residual = np.log(frame["y"]) - alpha - beta * np.log(frame["x"])
    residual.name = f"spread_{y.name}_{x.name}"
    return alpha, beta, residual


def build_spread(y: pd.Series, x: pd.Series, alpha: float, hedge_ratio: float) -> pd.Series:
    """Constructs a log-spread time series given fixed cointegration parameters.

    Calculates the spread using fixed intercept and hedge ratio parameters:
    spread = ln(y) - α - hedge_ratio · ln(x).

    Args:
        y (pd.Series): Price series for the dependent asset.
        x (pd.Series): Price series for the independent asset.
        alpha (float): Intercept parameter of the cointegrating relationship.
        hedge_ratio (float): Hedge ratio ($\beta$) parameter of the cointegrating relationship.

    Returns:
        pd.Series: Log-spread time series indexed by observation dates.
    """
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    spread = np.log(frame["y"]) - alpha - hedge_ratio * np.log(frame["x"])
    spread.name = f"spread_{y.name}_{x.name}"
    return spread


def _adf_pvalue(series: pd.Series, regression: str = "n") -> tuple[float, float, int]:
    """Runs an Augmented Dickey-Fuller (ADF) test to evaluate stationarity.

    Args:
        series (pd.Series): Time series data to test for unit roots.
        regression (str, optional): Constant and trend order for ADF test
            ('c' for constant, 'ct' for constant and trend, 'n' for no constant/trend).
            Defaults to "n".

    Returns:
        tuple[float, float, int]: A tuple containing:
            - **stat** (float): ADF test statistic. Returns `np.nan` if insufficient valid observations.
            - **pvalue** (float): MacKinnon's approximate p-value. Returns `np.nan` if insufficient valid observations.
            - **used_lag** (int): Number of lags included according to AIC criterion. Returns 0 if invalid.
    """
    values = pd.Series(series).dropna()
    if len(values) < 60 or values.nunique() < 5:
        return np.nan, np.nan, 0
    stat, pvalue, used_lag, *_ = adfuller(values, regression=regression, autolag="AIC")
    return float(stat), float(pvalue), int(used_lag)


def _i1_diagnostic(log_price: pd.Series) -> tuple[float, float, bool]:
    """Evaluates whether a log-price time series is plausibly integrated of order 1, I(1).

    Checks whether the price level is non-stationary (p > 0.05 with trend) and
    its first difference is stationary (p < 0.05 with constant).

    Args:
        log_price (pd.Series): Logarithm of asset price series.

    Returns:
        tuple[float, float, bool]: A tuple containing:
            - **level_p** (float): ADF p-value on log-price levels with trend.
            - **diff_p** (float): ADF p-value on first-differenced log-prices with constant.
            - **plausible** (bool): True if level is non-stationary and first difference is stationary.
    """
    _, level_p, _ = _adf_pvalue(log_price, regression="ct")
    _, diff_p, _ = _adf_pvalue(log_price.diff().dropna(), regression="c")
    plausible = bool(np.isfinite(level_p) and np.isfinite(diff_p) and level_p > 0.05 and diff_p < 0.05)
    return level_p, diff_p, plausible


def screen_pairs(training_prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Screen pairs only on the training/pair-selection window.

    The function reports all tests, applies Benjamini-Hochberg FDR to Engle-Granger p-values,
    and conservatively requires both Engle-Granger and residual-ADF diagnostics when configured.

    Tests all unique pair combinations for correlation, cointegration (Engle-Granger and
    residual ADF diagnostics), I(1) price integration, False Discovery Rate (FDR)
    p-value control via Benjamini-Hochberg, and mean-reversion half-life.

    Args:
        training_prices (pd.DataFrame): DataFrame of historical asset price series for screening.
        cfg (dict): Configuration dictionary containing settings under the `"pair_selection"` key
            (e.g., `min_abs_corr`, `max_eg_pvalue`, `max_residual_adf_pvalue`, `fdr_alpha`, `require_i1`,
            `require_both_cointegration_diagnostics`, `allow_best_available`, and `top_n`).

    Returns:
        tuple[pd.DataFrame, dict]: A tuple containing:
            - **selected_table** (pd.DataFrame): DataFrame of candidate pairs sorted by test pass
              status and custom screening score, containing statistical metrics and filter passes.
            - **summary** (dict): Dictionary with pair-screening execution metadata (e.g., total pairs tested,
              number passing all filters, and selected pair names).

    Raises:
        RuntimeError: If no candidate pairs meet observation length criteria, or if no pair
            passes all filters when `allow_best_available` is set to False.
    """
    pcfg = cfg["pair_selection"]
    returns = training_prices.pct_change().dropna()
    rows: list[dict] = []
    for asset_a, asset_b in combinations(training_prices.columns, 2):
        pair = training_prices[[asset_a, asset_b]].dropna()
        if len(pair) < 120:
            continue
        corr = float(returns[asset_a].corr(returns[asset_b]))
        try:
            alpha, beta, residual = estimate_cointegrating_regression(pair[asset_a], pair[asset_b])
            eg_stat, eg_p, _ = coint(
                np.log(pair[asset_a]),
                np.log(pair[asset_b]),
                trend="c",
                autolag="aic",
            )
            residual_adf_stat, residual_adf_p, residual_adf_lag = _adf_pvalue(residual, regression="n")
            a_level_p, a_diff_p, a_i1 = _i1_diagnostic(np.log(pair[asset_a]))
            b_level_p, b_diff_p, b_i1 = _i1_diagnostic(np.log(pair[asset_b]))
            half_life = estimate_half_life(residual)
        except Exception:
            alpha = beta = eg_stat = eg_p = residual_adf_stat = residual_adf_p = np.nan
            residual_adf_lag = 0
            a_level_p = a_diff_p = b_level_p = b_diff_p = np.nan
            a_i1 = b_i1 = False
            half_life = np.nan
        rows.append(
            {
                "asset_a": asset_a,
                "asset_b": asset_b,
                "abs_return_corr": abs(corr),
                "return_corr": corr,
                "alpha": alpha,
                "hedge_ratio": beta,
                "eg_stat": float(eg_stat) if np.isfinite(eg_stat) else np.nan,
                "eg_pvalue": float(eg_p) if np.isfinite(eg_p) else np.nan,
                "residual_adf_stat": residual_adf_stat,
                "residual_adf_pvalue": residual_adf_p,
                "residual_adf_lag": residual_adf_lag,
                "asset_a_level_adf_pvalue": a_level_p,
                "asset_a_diff_adf_pvalue": a_diff_p,
                "asset_b_level_adf_pvalue": b_level_p,
                "asset_b_diff_adf_pvalue": b_diff_p,
                "asset_a_plausibly_i1": a_i1,
                "asset_b_plausibly_i1": b_i1,
                "estimated_half_life_days": half_life,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No pairs had enough observations for screening.")
    finite = table["eg_pvalue"].fillna(1.0).to_numpy()
    reject, adjusted, _, _ = multipletests(finite, alpha=float(pcfg.get("fdr_alpha", 0.10)), method="fdr_bh")
    table["eg_fdr_pvalue"] = adjusted
    table["eg_fdr_reject"] = reject

    corr_pass = table["abs_return_corr"] >= float(pcfg["min_abs_corr"])
    eg_pass = table["eg_pvalue"] <= float(pcfg["max_eg_pvalue"])
    adf_pass = table["residual_adf_pvalue"] <= float(pcfg["max_residual_adf_pvalue"])
    fdr_pass = table["eg_fdr_reject"]
    i1_pass = table["asset_a_plausibly_i1"] & table["asset_b_plausibly_i1"]
    if not bool(pcfg.get("require_i1", True)):
        i1_pass = pd.Series(True, index=table.index)
    diagnostic_pass = eg_pass & adf_pass if bool(pcfg.get("require_both_cointegration_diagnostics", True)) else eg_pass
    table["passes_corr"] = corr_pass
    table["passes_eg"] = eg_pass
    table["passes_residual_adf"] = adf_pass
    table["passes_i1"] = i1_pass
    table["passes_all"] = corr_pass & diagnostic_pass & fdr_pass & i1_pass

    table["screen_score"] = (
        table["abs_return_corr"].fillna(0.0)
        - 0.30 * table["eg_fdr_pvalue"].fillna(1.0)
        - 0.20 * table["residual_adf_pvalue"].fillna(1.0)
        - 0.01 * table["estimated_half_life_days"].clip(lower=0, upper=100).fillna(100)
    )
    table = table.sort_values(["passes_all", "screen_score"], ascending=[False, False]).reset_index(drop=True)
    passing = table[table["passes_all"]].copy()
    selection_status = "passed_all_conservative_filters"
    if passing.empty:
        if not bool(pcfg.get("allow_best_available", True)):
            raise RuntimeError("No candidate pair passed all configured tests.")
        passing = table.head(1).copy()
        selection_status = "provisional_best_available_failed_one_or_more_filters"
    top_n = int(pcfg.get("top_n", 10))
    selected_table = pd.concat([passing.head(top_n), table[~table.index.isin(passing.index)].head(max(0, top_n - len(passing.head(top_n))))])
    selected_table = selected_table.drop_duplicates(["asset_a", "asset_b"]).head(top_n).reset_index(drop=True)
    summary = {
        "pairs_tested": int(len(table)),
        "pairs_passing_all": int(table["passes_all"].sum()),
        "selection_status": selection_status,
        "selected_asset_a": str(passing.iloc[0]["asset_a"]),
        "selected_asset_b": str(passing.iloc[0]["asset_b"]),
    }
    return selected_table, summary


def estimate_half_life(spread: pd.Series) -> float:
    """Estimates the mean-reversion half-life of a spread series using an AR(1) process.

    Fits an OLS regression on spread increments Δsₜ = α + β·sₜ₋₁ + εₜ
    and calculates the half-life as -ln(2)/β.

    Args:
        spread (pd.Series): Time series of spread residuals.

    Returns:
        float: Estimated half-life in periods (days). Returns `np.inf` if β >= 0
            (non-mean-reverting series) or `np.nan` if sample size < 60.
    """
    values = pd.Series(spread).dropna()
    if len(values) < 60:
        return np.nan
    lagged = values.shift(1).dropna()
    delta = values.diff().dropna().loc[lagged.index]
    fit = OLS(delta, add_constant(lagged)).fit()
    beta = float(fit.params.iloc[1])
    if beta >= 0:
        return np.inf
    return float(-np.log(2.0) / beta)


def select_top_pair(pair_table: pd.DataFrame) -> tuple[str, str, float, float, str]:
    """Selects the highest-ranked asset pair from the screened pair table.

    Args:
        pair_table (pd.DataFrame): Sorted DataFrame of candidate pairs output by `screen_pairs`.

    Returns:
        tuple[str, str, float, float, str]: A tuple containing:
            - **asset_a** (str): Primary asset ticker symbol.
            - **asset_b** (str): Secondary asset ticker symbol.
            - **alpha** (float): Cointegration intercept parameter.
            - **hedge_ratio** (float): Cointegration hedge ratio ($\beta$).
            - **status** (str): Selection qualification status (`"passed_all"` or `"provisional"`).
    """
    row = pair_table.iloc[0]
    status = "passed_all" if bool(row["passes_all"]) else "provisional"
    return str(row["asset_a"]), str(row["asset_b"]), float(row["alpha"]), float(row["hedge_ratio"]), status
