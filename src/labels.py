from __future__ import annotations

import numpy as np
import pandas as pd


def build_convergence_labels(
    signal_df: pd.DataFrame,
    exit_z: float = 0.5,
    max_holding_days: int = 20,
    transaction_cost_proxy: float = 0.0,
) -> pd.DataFrame:
    """Label each trade candidate by profitable convergence after costs.

    A positive label means a signal both converges toward the exit threshold and
    produces positive spread PnL over the forward window after a transaction-cost
    proxy. This is intentionally more conservative than a pure convergence label
    and usually creates both accepted and rejected examples for ML training.
    """
    df = signal_df.copy()
    z = df["zscore"].values
    spread = df["spread"].values
    raw_signal = df["raw_signal"].values
    labels = np.zeros(len(df), dtype=int)
    forward_min_abs_z = np.full(len(df), np.nan)
    forward_pnl = np.full(len(df), np.nan)
    for i in range(len(df)):
        end = min(i + max_holding_days + 1, len(df))
        if i + 1 >= end:
            continue
        future_abs = np.abs(z[i + 1:end])
        forward_min_abs_z[i] = np.nanmin(future_abs)
        horizon_idx = end - 1
        # long spread gains when spread rises; short spread gains when spread falls
        pnl = raw_signal[i] * (spread[horizon_idx] - spread[i])
        pnl -= transaction_cost_proxy
        forward_pnl[i] = pnl
        converged = np.nanmin(future_abs) <= exit_z
        labels[i] = int(converged and pnl > 0)
    df["forward_min_abs_z"] = forward_min_abs_z
    df["forward_pnl_proxy"] = forward_pnl
    df["converged_label"] = labels
    df["is_trade_candidate"] = (df["raw_signal"].abs() > 0).astype(int)
    return df


def get_model_dataset(labeled_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [
        "abs_zscore",
        "spread_volatility",
        "rolling_corr",
        "pair_return_diff",
        "drawdown_a",
        "drawdown_b",
        "half_life",
    ]
    data = labeled_df[labeled_df["is_trade_candidate"] == 1].dropna(subset=feature_cols + ["converged_label"])
    X = data[feature_cols]
    y = data["converged_label"].astype(int)
    return X, y
