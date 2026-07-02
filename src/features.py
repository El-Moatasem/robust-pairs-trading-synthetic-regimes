from __future__ import annotations

import numpy as np
import pandas as pd
from .diagnostics import estimate_hedge_ratio


def rolling_zscore(series: pd.Series, window: int = 30) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def estimate_half_life(spread: pd.Series) -> float:
    """Estimate mean-reversion half-life using AR(1)-style regression."""
    s = spread.dropna()
    if len(s) < 60:
        return np.nan
    lagged = s.shift(1).dropna()
    delta = s.diff().dropna()
    aligned = pd.concat([lagged, delta], axis=1).dropna()
    if aligned.empty:
        return np.nan
    x = aligned.iloc[:, 0].values
    y = aligned.iloc[:, 1].values
    beta = np.polyfit(x, y, 1)[0]
    if beta >= 0:
        return np.inf
    return float(-np.log(2) / beta)


def construct_pair_features(
    prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    zscore_window: int = 30,
    volatility_window: int = 30,
    correlation_window: int = 60,
) -> pd.DataFrame:
    """Create pair-level spread and risk/regime features."""
    y = prices[asset_a]
    x = prices[asset_b]
    hedge_ratio = estimate_hedge_ratio(y, x)
    spread = y - hedge_ratio * x
    returns_a = y.pct_change()
    returns_b = x.pct_change()
    z = rolling_zscore(spread, zscore_window)
    spread_ret = spread.diff()
    features = pd.DataFrame(index=prices.index)
    features["asset_a"] = asset_a
    features["asset_b"] = asset_b
    features["price_a"] = y
    features["price_b"] = x
    features["hedge_ratio"] = hedge_ratio
    features["spread"] = spread
    features["spread_change"] = spread_ret
    features["zscore"] = z
    features["abs_zscore"] = z.abs()
    features["spread_volatility"] = spread_ret.rolling(volatility_window).std()
    features["rolling_corr"] = returns_a.rolling(correlation_window).corr(returns_b)
    features["pair_return_diff"] = returns_a - returns_b
    features["drawdown_a"] = y / y.cummax() - 1
    features["drawdown_b"] = x / x.cummax() - 1
    features["half_life"] = estimate_half_life(spread)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def build_features_for_pairs(prices: pd.DataFrame, pairs: pd.DataFrame, **kwargs) -> pd.DataFrame:
    frames = []
    for _, row in pairs.iterrows():
        f = construct_pair_features(prices, row["asset_a"], row["asset_b"], **kwargs)
        frames.append(f)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames).sort_index()
