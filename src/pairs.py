from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, coint


def estimate_cointegrating_regression(y: pd.Series, x: pd.Series) -> tuple[float, float, pd.Series]:
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
    frame = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    spread = np.log(frame["y"]) - alpha - hedge_ratio * np.log(frame["x"])
    spread.name = f"spread_{y.name}_{x.name}"
    return spread


def _adf_pvalue(series: pd.Series, regression: str = "n") -> tuple[float, float, int]:
    values = pd.Series(series).dropna()
    if len(values) < 60 or values.nunique() < 5:
        return np.nan, np.nan, 0
    stat, pvalue, used_lag, *_ = adfuller(values, regression=regression, autolag="AIC")
    return float(stat), float(pvalue), int(used_lag)


def _i1_diagnostic(log_price: pd.Series) -> tuple[float, float, bool]:
    _, level_p, _ = _adf_pvalue(log_price, regression="ct")
    _, diff_p, _ = _adf_pvalue(log_price.diff().dropna(), regression="c")
    plausible = bool(np.isfinite(level_p) and np.isfinite(diff_p) and level_p > 0.05 and diff_p < 0.05)
    return level_p, diff_p, plausible


def screen_pairs(training_prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Screen pairs only on the training/pair-selection window.

    The function reports all tests, applies Benjamini-Hochberg FDR to Engle-Granger p-values,
    and conservatively requires both Engle-Granger and residual-ADF diagnostics when configured.
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
    row = pair_table.iloc[0]
    status = "passed_all" if bool(row["passes_all"]) else "provisional"
    return str(row["asset_a"]), str(row["asset_b"]), float(row["alpha"]), float(row["hedge_ratio"]), status
