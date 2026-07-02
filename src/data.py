from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional
import numpy as np
import pandas as pd


@dataclass
class DataConfig:
    tickers: list[str]
    start_date: str = "2018-01-01"
    end_date: str = "2025-12-31"
    price_field: str = "Adj Close"
    use_sample_data: bool = True
    random_state: int = 42


def generate_sample_prices(
    tickers: Iterable[str],
    n_days: int = 900,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate reproducible sample prices with related pairs.

    This function lets reviewers run the project without internet access.
    The generated panel intentionally contains related assets so that pair
    selection and cointegration-style diagnostics have meaningful inputs.
    """
    rng = np.random.default_rng(random_state)
    tickers = list(tickers)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    market = rng.normal(0.00025, 0.009, size=n_days).cumsum()
    prices = {}

    for idx, ticker in enumerate(tickers):
        pair_group = idx // 2
        common = market + rng.normal(0, 0.003 + 0.001 * pair_group, size=n_days).cumsum()
        idio = rng.normal(0, 0.006, size=n_days).cumsum()
        log_price = np.log(80 + 5 * idx) + common + 0.35 * idio
        if idx % 2 == 1:
            # Make every second ticker related to the previous ticker.
            prev = np.log(prices[tickers[idx - 1]])
            log_price = 0.92 * prev + 0.08 * log_price + rng.normal(0, 0.003, size=n_days)
        prices[ticker] = np.exp(log_price)

    return pd.DataFrame(prices, index=dates).round(4)


def load_prices_from_csv(path: str | Path) -> pd.DataFrame:
    """Load a CSV price panel whose index column is a date."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return clean_price_panel(df)


def download_prices_yfinance(config: DataConfig) -> pd.DataFrame:
    """Download adjusted close prices with yfinance.

    The function imports yfinance lazily so the package can still be imported
    on machines without yfinance installed.
    """
    import yfinance as yf

    data = yf.download(
        tickers=config.tickers,
        start=config.start_date,
        end=config.end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if isinstance(data.columns, pd.MultiIndex):
        if config.price_field in data.columns.get_level_values(0):
            prices = data[config.price_field]
        elif "Close" in data.columns.get_level_values(0):
            prices = data["Close"]
        else:
            raise ValueError(f"Could not find {config.price_field} or Close in downloaded data")
    else:
        prices = data[[config.price_field]] if config.price_field in data.columns else data[["Close"]]
    return clean_price_panel(prices)


def clean_price_panel(prices: pd.DataFrame) -> pd.DataFrame:
    """Clean price panel by sorting, de-duplicating, and forward-filling."""
    prices = prices.copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    prices = prices.loc[~prices.index.duplicated(keep="last")]
    prices = prices.apply(pd.to_numeric, errors="coerce")
    prices = prices.ffill().dropna(axis=1, how="all").dropna(how="all")
    prices = prices.dropna(axis=1, thresh=max(20, int(0.7 * len(prices))))
    prices = prices.ffill().bfill()
    return prices


def get_price_data(config: DataConfig, csv_path: Optional[str | Path] = None) -> pd.DataFrame:
    """Get price data from CSV, yfinance, or generated sample data."""
    if csv_path:
        return load_prices_from_csv(csv_path)
    if config.use_sample_data:
        return generate_sample_prices(config.tickers, random_state=config.random_state)
    return download_prices_yfinance(config)
