from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSummary:
    """Immutable data container holding execution metadata and dataset statistics.

    Attributes:
        source (str): Identifier or path representing the data source
            (e.g., 'yfinance_auto_adjusted_close', 'cached_public_adjusted_close',
            or 'offline_reproducible_cointegrated_synthetic_daily_prices').
        mode_requested (str): The requested operational data mode ('synthetic', 'public', or 'auto').
        is_synthetic (bool): Flag indicating whether the loaded dataset is synthetically generated.
        fallback_reason (str | None): Exception message or description if automatic fallback
            to synthetic data occurred; None otherwise.
        frequency (str): Sampling frequency of the price series (e.g., 'daily').
        n_assets (int): Total number of asset price series contained in the dataset.
        n_observations (int): Total number of time series observations/rows.
        start (str): Start date string of the time series dataset ('YYYY-MM-DD').
        end (str): End date string of the time series dataset ('YYYY-MM-DD').
    """
    source: str
    mode_requested: str
    is_synthetic: bool
    fallback_reason: str | None
    frequency: str
    n_assets: int
    n_observations: int
    start: str
    end: str

    def to_dict(self) -> dict:
        """Converts the DataSummary dataclass instance into a standard dictionary.

        Returns:
            dict: Dictionary representation of the dataset summary metadata.
        """
        return asdict(self)


def _ou_process(n: int, rng: np.random.Generator, phi: float, sigma: float, mean: float = 0.0) -> np.ndarray:
    """Generates a discrete-time Ornstein-Uhlenbeck (AR(1)) process time series.

    Computes a mean-reverting stochastic series starting from its stationary distribution.

    Args:
        n (int): Number of time steps/observations to generate.
        rng (np.random.Generator): NumPy random number generator for reproducible sampling.
        phi (float): Autoregressive coefficient controlling the rate of mean reversion (|phi| < 1).
        sigma (float): Standard deviation of the Gaussian innovation noise term.
        mean (float, optional): Long-term mean level of the process. Defaults to 0.0.

    Returns:
        np.ndarray: 1D array of shape `(n,)` containing the generated OU process values.
    """
    values = np.zeros(n, dtype=float)
    values[0] = mean + rng.normal(0.0, sigma / max(np.sqrt(1.0 - phi**2), 1e-6))
    for i in range(1, n):
        values[i] = mean + phi * (values[i - 1] - mean) + rng.normal(0.0, sigma)
    return values


def generate_synthetic_prices(
    tickers: Iterable[str],
    n_days: int = 1800,
    seed: int = 42,
    start: str = "2018-01-01",
) -> pd.DataFrame:
    """Generate deterministic daily prices with deliberately cointegrated pairs.

    Each designated pair shares an I(1) stochastic trend while its log-price residual follows
    a stationary AR(1)/OU process. This construction avoids the earlier generator's accidental
    random-walk residual, which could produce conflicting cointegration diagnostics.
    Generates deterministic daily asset prices with explicitly cointegrated pairs.

    Constructs synthetic log-price series using market, sector, and jump factors along
    with stochastic volatility. Pairs share an $I(1)$ stochastic trend with stationary
    Ornstein-Uhlenbeck residuals to guarantee stationarity without random-walk noise.

    Args:
        tickers (Iterable[str]): List or iterable of ticker symbols to generate prices for.
        n_days (int, optional): Total number of business trading days to simulate. Defaults to 1800.
        seed (int, optional): Random seed for NumPy random number generator. Defaults to 42.
        start (str, optional): Start date string for the business day date index formatted as
            'YYYY-MM-DD'. Defaults to "2018-01-01".

    Returns:
        pd.DataFrame: DataFrame containing simulated daily asset prices rounded to 4 decimal places,
            indexed by date (`pd.DatetimeIndex`) with asset tickers as column names.
    """
    rng = np.random.default_rng(seed)
    tickers = list(tickers)
    dates = pd.bdate_range(start=start, periods=n_days)

    market = rng.normal(0.00025, 0.009, n_days)
    sector_energy = rng.normal(0.00002, 0.006, n_days)
    sector_consumer = rng.normal(0.00002, 0.005, n_days)
    sector_finance = rng.normal(0.00002, 0.006, n_days)
    sector_tech = rng.normal(0.00004, 0.008, n_days)
    volatility_state = np.clip(_ou_process(n_days, rng, phi=0.96, sigma=0.08, mean=1.0), 0.45, 2.5)
    jumps = (rng.random(n_days) < 0.008) * rng.normal(0.0, 0.035, n_days)

    starts = {
        "KO": 60.0, "PEP": 170.0, "XOM": 110.0, "CVX": 160.0,
        "JPM": 145.0, "BAC": 35.0, "MSFT": 380.0, "AAPL": 190.0,
        "SPY": 500.0, "QQQ": 430.0,
    }
    sector_map = {
        "KO": sector_consumer, "PEP": sector_consumer,
        "XOM": sector_energy, "CVX": sector_energy,
        "JPM": sector_finance, "BAC": sector_finance,
        "MSFT": sector_tech, "AAPL": sector_tech, "QQQ": sector_tech,
        "SPY": np.zeros(n_days),
    }

    log_prices: dict[str, np.ndarray] = {}
    for idx, ticker in enumerate(tickers):
        idiosyncratic = rng.normal(0.0, 0.006 * volatility_state, n_days)
        daily_return = 0.0001 + 0.72 * market + 0.35 * sector_map.get(ticker, 0.0) + 0.28 * idiosyncratic + 0.08 * jumps
        log_prices[ticker] = np.log(starts.get(ticker, 100.0 + idx * 10.0)) + np.cumsum(daily_return)

    pair_specs = {
        ("KO", "PEP"): (1.02, 0.965, 0.0040),
        ("XOM", "CVX"): (0.98, 0.955, 0.0050),
        ("JPM", "BAC"): (0.92, 0.945, 0.0060),
    }
    for (left, right), (beta, phi, sigma) in pair_specs.items():
        if left not in log_prices or right not in log_prices:
            continue
        residual = _ou_process(n_days, rng, phi=phi, sigma=sigma)
        target_initial = np.log(starts[right])
        alpha = target_initial - beta * log_prices[left][0] - residual[0]
        log_prices[right] = alpha + beta * log_prices[left] + residual

    prices = pd.DataFrame({ticker: np.exp(log_prices[ticker]) for ticker in tickers}, index=dates)
    prices.index.name = "date"
    return prices.round(4)


def _download_public_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Downloads auto-adjusted daily closing market prices from Yahoo Finance.

    Cleans raw market data by forward-filling missing values and removing completely
    empty columns or rows.

    Args:
        tickers (list[str]): List of asset ticker symbols to download.
        start (str): Start date string for the data fetch formatted as 'YYYY-MM-DD'.
        end (str): End date string for the data fetch formatted as 'YYYY-MM-DD'.

    Returns:
        pd.DataFrame: Cleaned DataFrame of auto-adjusted Close prices, indexed by trading dates.

    Raises:
        RuntimeError: If `yfinance` returns empty data, missing Close price columns, or if
            the downloaded dataset contains fewer than 2 valid assets or 300 rows.
    """
    import yfinance as yf  # type: ignore

    downloaded = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=False,
    )
    if downloaded.empty:
        raise RuntimeError("yfinance returned no observations.")
    if isinstance(downloaded.columns, pd.MultiIndex):
        if "Close" not in downloaded.columns.get_level_values(0):
            raise RuntimeError("Downloaded data did not contain adjusted/auto-adjusted Close prices.")
        prices = downloaded["Close"].copy()
    else:
        if "Close" not in downloaded.columns:
            raise RuntimeError("Downloaded data did not contain a Close column.")
        prices = downloaded[["Close"]].rename(columns={"Close": tickers[0]})
    prices = prices.sort_index().ffill().dropna(how="all").dropna(axis=1, how="all")
    prices = prices.dropna()
    if prices.shape[1] < 2 or len(prices) < 300:
        raise RuntimeError(f"Public dataset is too small: {prices.shape[1]} assets and {len(prices)} rows.")
    return prices


def load_prices(config: dict, require_public: bool = False) -> tuple[pd.DataFrame, DataSummary]:
    """Loads asset price data based on configuration specifications with automatic fallback support.

    Attempts to load public historical asset prices from a cached CSV file or by downloading
    via Yahoo Finance. If public data fetch fails or is unavailable, it gracefully falls back
    to generating reproducible synthetic cointegrated daily price data.

    Args:
        config (dict): Configuration dictionary containing `data` specifications (e.g., `mode`,
            `tickers`, `frequency`, `start`, `end`, `cache_csv`, `fallback_days`) and `project` parameters.
        require_public (bool, optional): If True, forces requested mode to 'public' regardless of
            the configuration file setting. Defaults to False.

    Returns:
        tuple[pd.DataFrame, DataSummary]: A tuple containing:
            - **prices** (pd.DataFrame): DataFrame of daily price time series for specified assets.
            - **summary** (DataSummary): Dataclass containing execution metadata, data source details,
              and observation metrics.

    Raises:
        ValueError: If `config["data"]["mode"]` is not one of `synthetic`, `public`, or `auto`.
        RuntimeError: If `public` mode is strictly required but downloads/cache fail to load.
    """
    data_cfg = config["data"]
    requested_mode = str(data_cfg.get("mode", "synthetic")).lower()
    if require_public:
        requested_mode = "public"
    if requested_mode not in {"synthetic", "public", "auto"}:
        raise ValueError("data.mode must be one of: synthetic, public, auto")

    tickers = list(data_cfg["tickers"])
    frequency = str(data_cfg.get("frequency", "daily"))
    fallback_reason: str | None = None

    if requested_mode in {"public", "auto"}:
        cache_path = Path(data_cfg.get("cache_csv", "data/raw/public_prices.csv"))
        try:
            if cache_path.exists():
                prices = pd.read_csv(cache_path, index_col=0, parse_dates=True).sort_index().dropna()
                if prices.shape[1] < 2 or len(prices) < 300:
                    raise RuntimeError("Cached public dataset was too small.")
                source = f"cached_public_adjusted_close:{cache_path}"
            else:
                prices = _download_public_prices(tickers, str(data_cfg["start"]), str(data_cfg["end"]))
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                prices.to_csv(cache_path)
                source = "yfinance_auto_adjusted_close"
            summary = DataSummary(
                source=source,
                mode_requested=requested_mode,
                is_synthetic=False,
                fallback_reason=None,
                frequency=frequency,
                n_assets=prices.shape[1],
                n_observations=len(prices),
                start=str(prices.index.min().date()),
                end=str(prices.index.max().date()),
            )
            return prices, summary
        except Exception as exc:
            if requested_mode == "public":
                raise RuntimeError(
                    "Public-data mode was requested, but the download/cache could not be used. "
                    "Check internet access or place a clean CSV at data.cache_csv."
                ) from exc
            fallback_reason = f"Public data unavailable: {type(exc).__name__}: {exc}"

    prices = generate_synthetic_prices(
        tickers=tickers,
        n_days=int(data_cfg.get("fallback_days", 1800)),
        seed=int(config["project"].get("random_seed", 42)),
        start=str(data_cfg.get("start", "2018-01-01")),
    )
    summary = DataSummary(
        source="offline_reproducible_cointegrated_synthetic_daily_prices",
        mode_requested=requested_mode,
        is_synthetic=True,
        fallback_reason=fallback_reason,
        frequency=frequency,
        n_assets=prices.shape[1],
        n_observations=len(prices),
        start=str(prices.index.min().date()),
        end=str(prices.index.max().date()),
    )
    return prices, summary
