from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd
from .diagnostics import engle_granger_coint, phillips_ouliaris_proxy, estimate_hedge_ratio


def compute_return_correlation(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    return returns.corr()


def screen_candidate_pairs(
    prices: pd.DataFrame,
    min_abs_correlation: float = 0.70,
    max_cointegration_pvalue: float = 0.10,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank candidate pairs by correlation and cointegration diagnostics."""
    corr = compute_return_correlation(prices)
    rows = []
    for a, b in combinations(prices.columns, 2):
        abs_corr = abs(float(corr.loc[a, b]))
        if abs_corr < min_abs_correlation:
            continue
        eg = engle_granger_coint(prices[a], prices[b])
        po = phillips_ouliaris_proxy(prices[a], prices[b])
        beta = estimate_hedge_ratio(prices[a], prices[b])
        score = abs_corr - 0.25 * min(float(eg.get("pvalue", 1) or 1), 1.0)
        rows.append({
            "asset_a": a,
            "asset_b": b,
            "abs_return_corr": abs_corr,
            "eg_coint_pvalue": eg.get("pvalue"),
            "po_proxy_pvalue": po.get("residual_adf_pvalue"),
            "hedge_ratio": beta,
            "score": score,
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result = result.sort_values(["eg_coint_pvalue", "score"], ascending=[True, False])
    result = result[result["eg_coint_pvalue"].fillna(1) <= max_cointegration_pvalue]
    if result.empty:
        # If no pair passes the strict test, keep the best correlated pairs so the pipeline can continue.
        result = pd.DataFrame(rows).sort_values("score", ascending=False)
    return result.head(top_n).reset_index(drop=True)
