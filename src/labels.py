from __future__ import annotations

import numpy as np
import pandas as pd


def build_convergence_labels(features: pd.DataFrame, cfg: dict) -> pd.Series:
    lcfg = cfg["labels"]
    horizon = int(lcfg["horizon_days"])
    convergence_z = float(lcfg["convergence_zscore"])
    stop_loss_z = float(lcfg["stop_loss_zscore"])
    costs = float(lcfg.get("transaction_cost_bps", 0.0)) / 10000.0

    z = features["zscore"].copy()
    labels = pd.Series(index=features.index, dtype=int, name="accept_signal")
    for i in range(len(features) - horizon):
        current_z = z.iloc[i]
        future_abs = z.iloc[i + 1:i + horizon + 1].abs()
        converged = (future_abs <= convergence_z).any()
        stopped = (future_abs >= stop_loss_z).any()
        # Require a meaningful initial deviation and convergence before stop loss.
        economically_large = abs(current_z) >= 1.0 + 5 * costs
        labels.iloc[i] = int(economically_large and converged and not stopped)
    return labels.dropna().astype(int)
