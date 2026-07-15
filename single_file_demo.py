"""Minimal self-contained demonstration for quick review.

Run:
    python single_file_demo.py

This file avoids internet access and external project modules. It demonstrates the central idea:
construct a spread, compute a z-score, trade mean reversion, and print basic performance metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-01", periods=500)
    common = rng.normal(0.0002, 0.01, len(dates))
    a = 100 * np.exp(np.cumsum(common + rng.normal(0, 0.006, len(dates))))
    stationary_spread_noise = rng.normal(0, 0.004, len(dates))
    b = 80 * np.exp(np.cumsum(common + stationary_spread_noise))
    prices = pd.DataFrame({"Asset_A": a, "Asset_B": b}, index=dates)

    beta = np.cov(np.log(prices["Asset_A"]), np.log(prices["Asset_B"]))[0, 1] / np.var(np.log(prices["Asset_B"]))
    spread = np.log(prices["Asset_A"]) - beta * np.log(prices["Asset_B"])
    z = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()

    position = pd.Series(0.0, index=dates)
    current = 0.0
    for i in range(1, len(z)):
        if current == 0 and z.iloc[i] > 1.5:
            current = -1
        elif current == 0 and z.iloc[i] < -1.5:
            current = 1
        elif current != 0 and abs(z.iloc[i]) < 0.25:
            current = 0
        position.iloc[i] = current

    pnl = -(spread.diff().fillna(0) * position.shift(1).fillna(0))
    cum_pnl = pnl.cumsum()
    sharpe = np.sqrt(252) * pnl.mean() / (pnl.std(ddof=0) + 1e-12)
    max_dd = (cum_pnl - cum_pnl.cummax()).min()
    print("Single-file demo completed.")
    print(f"Estimated hedge ratio: {beta:.4f}")
    print(f"Cumulative PnL: {cum_pnl.iloc[-1]:.4f}")
    print(f"Sharpe ratio: {sharpe:.4f}")
    print(f"Max drawdown: {max_dd:.4f}")


if __name__ == "__main__":
    main()
