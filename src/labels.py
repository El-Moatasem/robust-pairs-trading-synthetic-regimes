from __future__ import annotations

import numpy as np
import pandas as pd


def build_convergence_labels(features: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create economic accept/reject labels only at valid entry-signal timestamps.

    A signal is accepted when the spread converges before hitting the stop within the configured
    horizon and the directionally signed spread return remains positive after estimated round-trip
    costs. Non-signal rows remain unlabeled and are excluded from model training.

    Args:
        features (pd.DataFrame): DataFrame containing feature time series. Must include
            `signal_zscore`, `contemporaneous_zscore_for_outcomes`, and `spread` columns.
        cfg (dict): Configuration dictionary containing label generation settings under
            the `"labels"` key (e.g., `horizon_days`, `entry_zscore`, `convergence_zscore`,
            `stop_loss_zscore`, `transaction_cost_bps_per_leg`, `slippage_bps_per_leg`,
            and `minimum_net_return`).

    Returns:
        pd.DataFrame: DataFrame containing labeled entry points with columns:
            - **accept_signal** (int): Binary target indicator (1 if trade converged with
              positive net return > minimum threshold, 0 otherwise).
            - **signal_direction** (float): Direction of the trade (-1.0 for short spread,
              1.0 for long spread).
            - **realized_holding_days** (int): Number of days held until convergence,
              stop-loss, or horizon expiration.
            - **realized_gross_return** (float): Unadjusted gross spread return.
            - **realized_net_return** (float): Net return after deducting 4-leg round-trip costs.
            - **label_exit_reason** (str): Cause of exit (`"convergence"`, `"stop_loss"`,
              or `"horizon"`).
    """
    lcfg = cfg["labels"]
    horizon = int(lcfg["horizon_days"])
    entry_z = float(lcfg["entry_zscore"])
    convergence_z = float(lcfg["convergence_zscore"])
    stop_z = float(lcfg["stop_loss_zscore"])
    per_leg_cost = (
        float(lcfg.get("transaction_cost_bps_per_leg", 0.0))
        + float(lcfg.get("slippage_bps_per_leg", 0.0))
    ) / 10000.0
    # A complete pair trade has two asset legs at entry and two at exit.
    round_trip_cost = 4.0 * per_leg_cost
    minimum_net_return = float(lcfg.get("minimum_net_return", 0.0))

    result = pd.DataFrame(
        index=features.index,
        columns=[
            "accept_signal",
            "signal_direction",
            "realized_holding_days",
            "realized_gross_return",
            "realized_net_return",
            "label_exit_reason",
        ],
    )
    signal_z = features["signal_zscore"]
    outcome_z = features["contemporaneous_zscore_for_outcomes"]
    spread = features["spread"]

    for i in range(len(features) - horizon):
        current_signal_z = float(signal_z.iloc[i])
        if not np.isfinite(current_signal_z) or abs(current_signal_z) < entry_z:
            continue
        direction = -float(np.sign(current_signal_z))
        entry_spread = float(spread.iloc[i])
        accepted = False
        exit_reason = "horizon"
        exit_offset = horizon
        gross_return = direction * (float(spread.iloc[i + horizon]) - entry_spread)
        for offset in range(1, horizon + 1):
            future_z = float(outcome_z.iloc[i + offset])
            gross = direction * (float(spread.iloc[i + offset]) - entry_spread)
            if abs(future_z) >= stop_z:
                exit_reason = "stop_loss"
                exit_offset = offset
                gross_return = gross
                break
            if abs(future_z) <= convergence_z:
                exit_reason = "convergence"
                exit_offset = offset
                gross_return = gross
                accepted = (gross - round_trip_cost) > minimum_net_return
                break
        net_return = gross_return - round_trip_cost
        result.iloc[i] = [
            int(accepted),
            direction,
            int(exit_offset),
            float(gross_return),
            float(net_return),
            exit_reason,
        ]

    numeric_cols = [
        "accept_signal",
        "signal_direction",
        "realized_holding_days",
        "realized_gross_return",
        "realized_net_return",
    ]
    for col in numeric_cols:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    labeled = result.dropna(subset=["accept_signal"]).copy()
    labeled["accept_signal"] = labeled["accept_signal"].astype(int)
    labeled["realized_holding_days"] = labeled["realized_holding_days"].astype(int)
    return labeled
