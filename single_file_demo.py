"""Self-contained demo for reviewers who want one copy/paste file.

This script generates sample prices, screens a pair, constructs a spread, runs a
z-score strategy, and prints basic metrics. It intentionally avoids internet
access and optional packages.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=600)
base = rng.normal(0.0002, 0.01, len(dates)).cumsum()
a = np.exp(np.log(100) + base + rng.normal(0, 0.005, len(dates)).cumsum())
b = np.exp(np.log(95) + 0.95 * base + rng.normal(0, 0.006, len(dates)).cumsum())
prices = pd.DataFrame({"Asset_A": a, "Asset_B": b}, index=dates)

beta = np.polyfit(prices["Asset_B"], prices["Asset_A"], 1)[0]
spread = prices["Asset_A"] - beta * prices["Asset_B"]
z = (spread - spread.rolling(30).mean()) / spread.rolling(30).std()
position = pd.Series(0, index=dates, dtype=float)
position[z <= -2] = 1
position[z >= 2] = -1
position = position.replace(0, np.nan).ffill().fillna(0)
position[z.abs() <= 0.5] = 0
position[z.abs() >= 3.5] = 0
pnl = position.shift(1).fillna(0) * spread.diff().fillna(0)

print("Pair: Asset_A / Asset_B")
print(f"Hedge ratio: {beta:.4f}")
print(f"Cumulative PnL: {pnl.sum():.4f}")
print(f"Sharpe: {np.sqrt(252) * pnl.mean() / pnl.std(ddof=0):.4f}")
print(f"Max drawdown: {(pnl.cumsum() - pnl.cumsum().cummax()).min():.4f}")
