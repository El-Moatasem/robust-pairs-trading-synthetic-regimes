from __future__ import annotations

import numpy as np
import pandas as pd


def replay_trade_backtest(
    features: pd.DataFrame,
    cfg: dict,
    accept_filter: pd.Series | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Independent trade-replay validation of the primary daily accounting engine.

    This implementation does not reuse ``src.backtest.run_backtest``.  It identifies entry/exit
    events from the lagged signal, then computes each closed trade directly from entry and exit
    spread levels.  Agreement of cumulative PnL and trade counts with the primary engine is a
    useful implementation cross-check.
    """
    bcfg = cfg["backtest"]
    entry_z = float(bcfg["entry_z"])
    exit_z = float(bcfg["exit_z"])
    stop_z = float(bcfg["stop_z"])
    max_holding = int(bcfg["max_holding_days"])
    per_leg = (
        float(bcfg.get("transaction_cost_bps_per_leg", 0.0))
        + float(bcfg.get("slippage_bps_per_leg", 0.0))
    ) / 10000.0
    round_trip_cost = 4.0 * per_leg

    frame = features.sort_index()
    accept = None if accept_filter is None else accept_filter.reindex(frame.index).fillna(0).astype(bool)
    open_trade: dict | None = None
    rows: list[dict] = []

    for i in range(1, len(frame)):
        date = frame.index[i]
        z = float(frame["signal_zscore"].iloc[i])
        spread = float(frame["spread"].iloc[i])
        allow_entry = True if accept is None else bool(accept.iloc[i])

        if open_trade is None:
            if allow_entry and abs(z) >= entry_z:
                open_trade = {
                    "entry_date": date,
                    "entry_spread": spread,
                    "entry_zscore": z,
                    "direction": -float(np.sign(z)),
                    "holding_days": 0,
                }
            continue

        open_trade["holding_days"] += 1
        reason = ""
        if abs(z) <= exit_z:
            reason = "convergence"
        elif abs(z) >= stop_z:
            reason = "stop_loss"
        elif int(open_trade["holding_days"]) >= max_holding:
            reason = "max_holding"
        if reason:
            gross = float(open_trade["direction"] * (spread - open_trade["entry_spread"]))
            rows.append(
                {
                    **open_trade,
                    "exit_date": date,
                    "exit_spread": spread,
                    "exit_reason": reason,
                    "gross_pnl": gross,
                    "cost": round_trip_cost,
                    "net_pnl": gross - round_trip_cost,
                }
            )
            open_trade = None

    if open_trade is not None and len(frame):
        date = frame.index[-1]
        spread = float(frame["spread"].iloc[-1])
        gross = float(open_trade["direction"] * (spread - open_trade["entry_spread"]))
        rows.append(
            {
                **open_trade,
                "exit_date": date,
                "exit_spread": spread,
                "exit_reason": "end_of_sample",
                "gross_pnl": gross,
                "cost": round_trip_cost,
                "net_pnl": gross - round_trip_cost,
            }
        )

    trades = pd.DataFrame(rows)
    if trades.empty:
        return trades, {"cumulative_net_pnl": 0.0, "closed_trades": 0, "trade_win_rate": 0.0}
    summary = {
        "cumulative_net_pnl": float(trades["net_pnl"].sum()),
        "closed_trades": int(len(trades)),
        "trade_win_rate": float((trades["net_pnl"] > 0).mean()),
        "average_holding_days": float(trades["holding_days"].mean()),
        "total_cost": float(trades["cost"].sum()),
    }
    return trades, summary


def compare_backtest_engines(primary_trades: pd.DataFrame, independent_trades: pd.DataFrame) -> dict:
    primary_pnl = float(primary_trades["net_pnl"].sum()) if len(primary_trades) else 0.0
    independent_pnl = float(independent_trades["net_pnl"].sum()) if len(independent_trades) else 0.0
    return {
        "primary_trade_count": int(len(primary_trades)),
        "independent_trade_count": int(len(independent_trades)),
        "primary_trade_pnl": primary_pnl,
        "independent_trade_pnl": independent_pnl,
        "absolute_pnl_difference": abs(primary_pnl - independent_pnl),
        "trade_count_match": bool(len(primary_trades) == len(independent_trades)),
        "pnl_match_within_1e_10": bool(abs(primary_pnl - independent_pnl) <= 1e-10),
    }
