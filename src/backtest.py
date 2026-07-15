from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_zscore(features: pd.DataFrame, cfg: dict, accept_filter: pd.Series | None = None) -> pd.DataFrame:
    bcfg = cfg["backtest"]
    entry_z = float(bcfg["entry_z"])
    exit_z = float(bcfg["exit_z"])
    stop_z = float(bcfg["stop_z"])
    max_holding = int(bcfg["max_holding_days"])
    cost = (float(bcfg.get("transaction_cost_bps", 0.0)) + float(bcfg.get("slippage_bps", 0.0))) / 10000.0

    df = features.copy()
    df["position"] = 0.0
    df["trade"] = 0.0
    df["pnl"] = 0.0
    position = 0.0
    holding = 0
    for i in range(1, len(df)):
        z = df["zscore"].iloc[i]
        allow = True if accept_filter is None else bool(accept_filter.reindex(df.index).fillna(0).iloc[i])
        new_position = position
        if position == 0 and allow:
            if z >= entry_z:
                new_position = -1.0
                holding = 0
            elif z <= -entry_z:
                new_position = 1.0
                holding = 0
        elif position != 0:
            holding += 1
            if abs(z) <= exit_z or abs(z) >= stop_z or holding >= max_holding:
                new_position = 0.0
                holding = 0
        trade = new_position - position
        spread_ret = -(df["spread"].iloc[i] - df["spread"].iloc[i - 1]) * position
        trading_cost = abs(trade) * cost
        df.iloc[i, df.columns.get_loc("position")] = new_position
        df.iloc[i, df.columns.get_loc("trade")] = trade
        df.iloc[i, df.columns.get_loc("pnl")] = spread_ret - trading_cost
        position = new_position
    df["cum_pnl"] = df["pnl"].cumsum()
    return df


def summarize_backtest(results: pd.DataFrame, name: str) -> dict:
    pnl = results["pnl"].dropna()
    ann_factor = 252
    sharpe = np.sqrt(ann_factor) * pnl.mean() / (pnl.std(ddof=0) + 1e-12)
    cum = results["cum_pnl"]
    drawdown = cum - cum.cummax()
    trades = results["trade"].abs().sum()
    wins = (pnl > 0).mean() if len(pnl) else 0
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    return {
        "strategy": name,
        "cumulative_pnl": float(cum.iloc[-1]),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "win_rate": float(wins),
        "turnover": float(trades),
        "profit_factor": float(gross_profit / (gross_loss + 1e-12)),
    }
