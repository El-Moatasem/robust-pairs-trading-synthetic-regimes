"""Minimal offline demonstration of a cointegrated spread and lagged z-score backtest.

Run:
    python single_file_demo.py

The demo is deliberately small. The full research controls are implemented in run_pipeline.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2023-01-02", periods=650)
    common_log_price = np.log(100.0) + np.cumsum(rng.normal(0.0002, 0.01, len(dates)))
    residual = np.zeros(len(dates))
    for i in range(1, len(dates)):
        residual[i] = 0.96 * residual[i - 1] + rng.normal(0.0, 0.005)
    log_b = common_log_price
    log_a = 0.20 + 0.95 * log_b + residual
    spread = pd.Series(log_a - 0.20 - 0.95 * log_b, index=dates)
    zscore = ((spread - spread.rolling(30).mean()) / spread.rolling(30).std(ddof=0)).shift(1)

    position = pd.Series(0.0, index=dates)
    current = 0.0
    for i in range(1, len(zscore)):
        z = zscore.iloc[i]
        if not np.isfinite(z):
            continue
        if current == 0.0 and abs(z) >= 1.5:
            current = -float(np.sign(z))
        elif current != 0.0 and (abs(z) <= 0.25 or abs(z) >= 3.0):
            current = 0.0
        position.iloc[i] = current

    pnl = position.shift(1).fillna(0.0) * spread.diff().fillna(0.0)
    cumulative = pnl.cumsum()
    sharpe = np.sqrt(252) * pnl.mean() / (pnl.std(ddof=0) + 1e-12)
    drawdown = cumulative - cumulative.cummax()
    print("Single-file demo completed.")
    print("Data source: offline synthetic cointegrated pair")
    print("Signal timing: one-day lag")
    print(f"Cumulative PnL: {cumulative.iloc[-1]:.4f}")
    print(f"Sharpe ratio: {sharpe:.4f}")
    print(f"Maximum drawdown: {drawdown.min():.4f}")


if __name__ == "__main__":
    main()
