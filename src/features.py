from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

from .pairs import build_spread


MODEL_FEATURE_COLUMNS = [
    "signal_zscore",
    "abs_zscore",
    "spread_change_lagged",
    "spread_volatility",
    "rolling_corr",
    "spread_drawdown",
    "half_life",
    "deviation_persistence",
    "regime_stress_proxy",
]


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def _rolling_half_life(spread: pd.Series, window: int) -> pd.Series:
    result = pd.Series(np.nan, index=spread.index, dtype=float)
    for i in range(window, len(spread) + 1):
        sample = spread.iloc[i - window:i].dropna()
        if len(sample) < max(40, window // 2):
            continue
        lagged = sample.shift(1).dropna()
        delta = sample.diff().dropna().loc[lagged.index]
        if lagged.var() <= 1e-12:
            continue
        beta = float(OLS(delta, add_constant(lagged)).fit().params.iloc[1])
        if beta < -1e-8:
            result.iloc[i - 1] = min(float(-np.log(2.0) / beta), 252.0)
    return result.ffill()


def _consecutive_deviation_count(abs_zscore: pd.Series, threshold: float) -> pd.Series:
    values = abs_zscore.fillna(0.0).to_numpy()
    counts = np.zeros(len(values), dtype=float)
    current = 0
    for i, value in enumerate(values):
        if value >= threshold:
            current += 1
        else:
            current = 0
        counts[i] = current
    return pd.Series(counts, index=abs_zscore.index, dtype=float)


def _rolling_percentile_last(series: pd.Series, window: int) -> pd.Series:
    def percentile(values: np.ndarray) -> float:
        if len(values) == 0 or not np.isfinite(values[-1]):
            return np.nan
        finite = values[np.isfinite(values)]
        if len(finite) < 10:
            return np.nan
        return float((finite <= values[-1]).mean())

    return series.rolling(window=window, min_periods=max(20, window // 4)).apply(percentile, raw=True)


def build_feature_frame(
    prices: pd.DataFrame,
    asset_a: str,
    asset_b: str,
    cfg: dict,
    alpha: float,
    hedge_ratio: float,
) -> pd.DataFrame:
    """Build a no-look-ahead feature frame using a fixed training-period hedge ratio.

    The raw spread is retained for realized PnL. Every predictor used by the ML model and trading
    signal is shifted by `lag_predictors_by_days`, so a decision timestamp never uses the same-day
    close that generates its subsequent PnL.
    """
    fcfg = cfg["features"]
    pair = prices[[asset_a, asset_b]].dropna().copy()
    spread = build_spread(pair[asset_a], pair[asset_b], alpha=alpha, hedge_ratio=hedge_ratio)
    returns_a = np.log(pair[asset_a]).diff()
    returns_b = np.log(pair[asset_b]).diff()

    raw_zscore = rolling_zscore(spread, int(fcfg["lookback_zscore"]))
    raw_spread_change = spread.diff()
    raw_volatility = raw_spread_change.rolling(
        int(fcfg["lookback_volatility"]), min_periods=int(fcfg["lookback_volatility"])
    ).std(ddof=0)
    raw_corr = returns_a.rolling(
        int(fcfg["lookback_correlation"]), min_periods=int(fcfg["lookback_correlation"])
    ).corr(returns_b)
    rolling_peak = spread.rolling(
        int(fcfg["lookback_drawdown"]), min_periods=int(fcfg["lookback_drawdown"])
    ).max()
    raw_drawdown = spread - rolling_peak
    raw_half_life = _rolling_half_life(spread, int(fcfg.get("half_life_window", 120)))
    raw_persistence = _consecutive_deviation_count(
        raw_zscore.abs(), float(fcfg.get("persistence_threshold_z", 1.0))
    )

    stress_window = int(fcfg.get("stress_window", 252))
    volatility_pct = _rolling_percentile_last(raw_volatility, stress_window)
    weakening_corr_pct = _rolling_percentile_last(-raw_corr, stress_window)
    raw_stress = 0.5 * volatility_pct + 0.5 * weakening_corr_pct

    lag = int(fcfg.get("lag_predictors_by_days", 1))
    frame = pd.DataFrame(index=pair.index)
    frame["asset_a_price"] = pair[asset_a]
    frame["asset_b_price"] = pair[asset_b]
    frame["alpha"] = alpha
    frame["hedge_ratio"] = hedge_ratio
    frame["spread"] = spread
    frame["contemporaneous_zscore_for_outcomes"] = raw_zscore
    frame["signal_zscore"] = raw_zscore.shift(lag)
    frame["abs_zscore"] = raw_zscore.abs().shift(lag)
    frame["spread_change_lagged"] = raw_spread_change.shift(lag)
    frame["spread_volatility"] = raw_volatility.shift(lag)
    frame["rolling_corr"] = raw_corr.shift(lag)
    frame["spread_drawdown"] = raw_drawdown.shift(lag)
    frame["half_life"] = raw_half_life.shift(lag)
    frame["deviation_persistence"] = raw_persistence.shift(lag)
    frame["regime_stress_proxy"] = raw_stress.shift(lag)
    frame["asset_a"] = asset_a
    frame["asset_b"] = asset_b
    return frame.replace([np.inf, -np.inf], np.nan).dropna(subset=MODEL_FEATURE_COLUMNS + ["spread"])
