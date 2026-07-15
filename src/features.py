from __future__ import annotations

import numpy as np
import pandas as pd

from .pairs import build_spread, estimate_hedge_ratio


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std(ddof=0)
    return (series - mu) / sigma.replace(0, np.nan)


def estimate_half_life(spread: pd.Series, lag: int = 1) -> pd.Series:
    # rolling approximation: regress delta spread on lagged spread, convert AR coefficient into half-life proxy.
    s = spread.dropna()
    out = pd.Series(index=spread.index, dtype=float)
    window = 60
    for i in range(window, len(s)):
        sample = s.iloc[i-window:i]
        y = sample.diff().dropna()
        x = sample.shift(lag).dropna().loc[y.index]
        if len(x) < 20 or np.var(x) == 0:
            continue
        beta = np.cov(y, x)[0, 1] / np.var(x)
        if beta >= 0:
            hl = np.nan
        else:
            hl = -np.log(2) / beta
        out.loc[s.index[i]] = hl
    return out.ffill()


def build_feature_frame(prices: pd.DataFrame, asset_a: str, asset_b: str, cfg: dict, hedge_ratio: float | None = None) -> pd.DataFrame:
    fcfg = cfg["features"]
    pair = prices[[asset_a, asset_b]].dropna()
    if hedge_ratio is None:
        hedge_ratio = estimate_hedge_ratio(pair[asset_a], pair[asset_b])
    spread = build_spread(pair[asset_a], pair[asset_b], hedge_ratio)
    returns_a = pair[asset_a].pct_change()
    returns_b = pair[asset_b].pct_change()
    features = pd.DataFrame(index=pair.index)
    features["asset_a"] = pair[asset_a]
    features["asset_b"] = pair[asset_b]
    features["hedge_ratio"] = hedge_ratio
    features["spread"] = spread
    features["zscore"] = rolling_zscore(spread, fcfg["lookback_zscore"])
    features["spread_change"] = spread.diff()
    features["spread_volatility"] = spread.diff().rolling(fcfg["lookback_volatility"]).std()
    features["rolling_corr"] = returns_a.rolling(fcfg["lookback_correlation"]).corr(returns_b)
    cumulative = spread - spread.rolling(fcfg["lookback_drawdown"]).max()
    features["spread_drawdown"] = cumulative
    features["half_life"] = estimate_half_life(spread, fcfg.get("half_life_lag", 1))
    features["abs_zscore"] = features["zscore"].abs()
    features["deviation_persistence"] = (features["abs_zscore"] > 1.0).rolling(10).mean()
    features["regime_stress_proxy"] = features["spread_volatility"].rank(pct=True) * (1 - features["rolling_corr"].rank(pct=True))
    return features.dropna()
