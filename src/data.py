from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class DataSummary:
    source: str
    frequency: str
    n_assets: int
    n_observations: int
    start: str
    end: str


def _simulate_pair(base_returns: np.ndarray, beta: float, spread_noise: np.ndarray, start_price: float) -> np.ndarray:
    # The second asset inherits the common trend plus a stationary spread disturbance.
    log_price = np.log(start_price) + np.cumsum(beta * base_returns + spread_noise)
    return np.exp(log_price)


def generate_synthetic_prices(
    tickers: Iterable[str],
    n_days: int = 900,
    seed: int = 42,
    start: str = "2021-01-01",
) -> pd.DataFrame:
    """Generate reproducible daily prices with several realistic pairs.

    The generator is intentionally deterministic so instructors can run the code without internet.
    It creates common factors, pair-level cointegrating relationships, volatility clustering, and
    occasional shocks. The output mimics adjusted close prices at daily frequency.
    """
    rng = np.random.default_rng(seed)
    tickers = list(tickers)
    dates = pd.bdate_range(start=start, periods=n_days)

    market = rng.normal(0.00025, 0.010, n_days)
    sector_1 = rng.normal(0.00005, 0.008, n_days)
    sector_2 = rng.normal(0.00003, 0.009, n_days)
    vol_state = np.abs(rng.normal(1.0, 0.25, n_days))
    shock = (rng.random(n_days) < 0.015) * rng.normal(0, 0.035, n_days)

    prices = {}
    base_start = {"KO": 60, "PEP": 170, "XOM": 110, "CVX": 160, "JPM": 145, "BAC": 35, "MSFT": 380, "AAPL": 190, "SPY": 500, "QQQ": 430}
    for idx, t in enumerate(tickers):
        drift = 0.00015 + 0.00002 * idx
        factor = market + (sector_1 if idx % 2 == 0 else sector_2)
        idio = rng.normal(0, 0.007 * vol_state, n_days)
        returns = drift + 0.65 * factor + 0.35 * idio + 0.10 * shock
        prices[t] = np.exp(np.log(base_start.get(t, 100 + 10 * idx)) + np.cumsum(returns))

    # Force several realistic high-correlation pairs with stationary deviations.
    if "KO" in tickers and "PEP" in tickers:
        spread = rng.normal(0, 0.004, n_days)
        prices["PEP"] = _simulate_pair(np.diff(np.r_[np.log(prices["KO"])[0], np.log(prices["KO"])]), 1.00, spread, base_start["PEP"])
    if "XOM" in tickers and "CVX" in tickers:
        spread = rng.normal(0, 0.006, n_days)
        prices["CVX"] = _simulate_pair(np.diff(np.r_[np.log(prices["XOM"])[0], np.log(prices["XOM"])]), 1.03, spread, base_start["CVX"])
    if "JPM" in tickers and "BAC" in tickers:
        spread = rng.normal(0, 0.008, n_days)
        prices["BAC"] = _simulate_pair(np.diff(np.r_[np.log(prices["JPM"])[0], np.log(prices["JPM"])]), 0.95, spread, base_start["BAC"])

    df = pd.DataFrame(prices, index=dates)
    df.index.name = "date"
    return df.round(4)


def load_public_or_synthetic_data(config: dict) -> tuple[pd.DataFrame, DataSummary]:
    data_cfg = config["data"]
    tickers = data_cfg["tickers"]
    use_yfinance = bool(data_cfg.get("use_yfinance", False))
    seed = config["project"].get("random_seed", 42)

    if use_yfinance:
        try:
            import yfinance as yf  # type: ignore

            downloaded = yf.download(tickers, start=data_cfg["start"], end=data_cfg["end"], auto_adjust=True, progress=False)
            if isinstance(downloaded.columns, pd.MultiIndex):
                prices = downloaded["Close"].dropna(how="all")
            else:
                prices = downloaded.rename(columns={"Close": tickers[0]})[[tickers[0]]]
            prices = prices.ffill().dropna(axis=1, how="all").dropna()
            if prices.shape[1] >= 2 and len(prices) > 200:
                summary = DataSummary("yfinance_adjusted_close", data_cfg["frequency"], prices.shape[1], len(prices), str(prices.index.min().date()), str(prices.index.max().date()))
                return prices, summary
        except Exception:
            pass

    prices = generate_synthetic_prices(tickers, data_cfg.get("fallback_days", 900), seed, data_cfg.get("start", "2021-01-01"))
    summary = DataSummary("offline_reproducible_synthetic_daily_prices", data_cfg["frequency"], prices.shape[1], len(prices), str(prices.index.min().date()), str(prices.index.max().date()))
    return prices, summary
