from __future__ import annotations

import numpy as np
import pandas as pd


def generate_baseline_signals(features: pd.DataFrame, entry_z: float = 2.0, exit_z: float = 0.5, stop_z: float = 3.5) -> pd.DataFrame:
    """Generate long/short spread positions from z-score rules."""
    df = features.copy()
    df["raw_signal"] = 0
    df.loc[df["zscore"] <= -entry_z, "raw_signal"] = 1   # long spread
    df.loc[df["zscore"] >= entry_z, "raw_signal"] = -1   # short spread
    df.loc[df["abs_zscore"] <= exit_z, "raw_signal"] = 0
    df.loc[df["abs_zscore"] >= stop_z, "raw_signal"] = 0
    df["position"] = df["raw_signal"].replace(0, np.nan).ffill().fillna(0)
    df.loc[df["abs_zscore"] <= exit_z, "position"] = 0
    df.loc[df["abs_zscore"] >= stop_z, "position"] = 0
    return df


def apply_ml_filter(signals: pd.DataFrame, probabilities: pd.Series, threshold: float = 0.55) -> pd.DataFrame:
    """Keep only trades whose predicted probability exceeds threshold."""
    df = signals.copy()
    aligned = probabilities.reindex(df.index).fillna(0)
    df["ml_probability"] = aligned
    df["ml_accept"] = (df["ml_probability"] >= threshold).astype(int)
    df["position"] = df["position"] * df["ml_accept"]
    return df


def backtest_pair_strategy(signals: pd.DataFrame, transaction_cost_bps: float = 5) -> tuple[pd.DataFrame, dict]:
    """Backtest spread strategy on generated positions.

    Returns both the enriched trade dataframe and summary metrics.
    """
    df = signals.copy()
    df["spread_return"] = df["spread"].diff().fillna(0)
    # Long spread gains when spread rises; short spread gains when spread falls.
    df["gross_pnl"] = df["position"].shift(1).fillna(0) * df["spread_return"]
    turnover = df["position"].diff().abs().fillna(0)
    cost = turnover * (transaction_cost_bps / 10000.0) * df["spread"].abs().rolling(20).mean().bfill()
    df["transaction_cost"] = cost.fillna(0)
    df["net_pnl"] = df["gross_pnl"] - df["transaction_cost"]
    df["equity_curve"] = df["net_pnl"].cumsum()
    metrics = summarize_backtest(df)
    return df, metrics


def summarize_backtest(df: pd.DataFrame) -> dict:
    pnl = df["net_pnl"].dropna()
    if len(pnl) == 0:
        return {}
    daily_mean = pnl.mean()
    daily_std = pnl.std(ddof=0)
    sharpe = np.nan if daily_std == 0 else float(np.sqrt(252) * daily_mean / daily_std)
    downside = pnl[pnl < 0].std(ddof=0)
    sortino = np.nan if downside == 0 or np.isnan(downside) else float(np.sqrt(252) * daily_mean / downside)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax()
    turnover = df["position"].diff().abs().sum()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = np.nan if losses.abs().sum() == 0 else float(wins.sum() / losses.abs().sum())
    return {
        "cumulative_pnl": float(pnl.sum()),
        "average_daily_pnl": float(daily_mean),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((pnl > 0).mean()),
        "turnover": float(turnover),
        "profit_factor": profit_factor,
        "n_observations": int(len(pnl)),
    }
