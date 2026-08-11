from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataSummary:
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
        return asdict(self)


def _ou_process(n: int, rng: np.random.Generator, phi: float, sigma: float, mean: float = 0.0) -> np.ndarray:
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
                prices = pd.read_csv(cache_path, index_col=0, parse_dates=True).sort_index()
                missing = [ticker for ticker in tickers if ticker not in prices.columns]
                if missing:
                    raise RuntimeError(
                        f"Cached public dataset does not contain requested tickers: {missing}. "
                        f"Delete {cache_path} to force a fresh download or provide a matching cache."
                    )
                prices = prices[tickers].loc[str(data_cfg["start"]):str(data_cfg["end"])].ffill().dropna()
                if prices.shape[1] < 2 or len(prices) < 300:
                    raise RuntimeError("Cached public dataset was too small after ticker/date validation.")
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
