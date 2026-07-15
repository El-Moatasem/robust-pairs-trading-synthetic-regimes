from __future__ import annotations

from itertools import combinations
from typing import Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    x_vals = np.asarray(x, dtype=float)
    y_vals = np.asarray(y, dtype=float)
    beta = np.cov(y_vals, x_vals)[0, 1] / np.var(x_vals)
    return float(beta)


def build_spread(y: pd.Series, x: pd.Series, hedge_ratio: float) -> pd.Series:
    spread = np.log(y) - hedge_ratio * np.log(x)
    return pd.Series(spread, index=y.index, name=f"spread_{y.name}_{x.name}")


def screen_pairs(prices: pd.DataFrame, min_abs_corr: float = 0.55, max_pvalue: float = 0.10, top_n: int = 8) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    rows = []
    for a, b in combinations(prices.columns, 2):
        aligned = prices[[a, b]].dropna()
        if len(aligned) < 100:
            continue
        corr = returns[a].corr(returns[b])
        try:
            coint_score, coint_p, _ = coint(np.log(aligned[a]), np.log(aligned[b]))
        except Exception:
            coint_p = np.nan
        beta = estimate_hedge_ratio(aligned[a], aligned[b])
        spread = build_spread(aligned[a], aligned[b], beta)
        try:
            adf_p = adfuller(spread.dropna())[1]
        except Exception:
            adf_p = np.nan
        score = abs(corr) - 0.2 * np.nan_to_num(coint_p, nan=1.0) - 0.1 * np.nan_to_num(adf_p, nan=1.0)
        rows.append({
            "asset_a": a,
            "asset_b": b,
            "abs_return_corr": abs(float(corr)),
            "eg_coint_pvalue": float(coint_p) if not np.isnan(coint_p) else np.nan,
            "spread_adf_pvalue": float(adf_p) if not np.isnan(adf_p) else np.nan,
            "hedge_ratio": beta,
            "score": score,
        })
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    filtered = df[(df["abs_return_corr"] >= min_abs_corr) & (df["eg_coint_pvalue"] <= max_pvalue)]
    if filtered.empty:
        filtered = df.head(top_n)
    return filtered.head(top_n).reset_index(drop=True)


def select_top_pair(pair_table: pd.DataFrame) -> Tuple[str, str, float]:
    row = pair_table.iloc[0]
    return str(row["asset_a"]), str(row["asset_b"]), float(row["hedge_ratio"])
