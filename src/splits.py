from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_index: pd.Index
    validation_index: pd.Index
    test_index: pd.Index
    pair_selection_end: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    purge_horizon_days: int
    embargo_days: int

    def summary(self) -> dict:
        result = asdict(self)
        result["train_index"] = len(self.train_index)
        result["validation_index"] = len(self.validation_index)
        result["test_index"] = len(self.test_index)
        return result


def _safe_date(index: pd.Index, position: int) -> str:
    value = index[position]
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def make_time_split(index: pd.Index, cfg: dict) -> TimeSplit:
    """Create chronological train/validation/test sets with purge and embargo gaps.

    Labels use future returns up to `purge_horizon_days`, so observations immediately before a
    boundary are removed from the earlier set. An additional embargo removes observations after
    each boundary from the next set. This prevents overlapping label horizons across partitions.
    """
    if len(index) < 150:
        raise ValueError("At least 150 observations are required for a purged three-way time split.")
    scfg = cfg["split"]
    pair_fraction = float(scfg.get("pair_selection_fraction", 0.50))
    val_fraction = float(scfg.get("validation_fraction", 0.20))
    test_fraction = float(scfg.get("test_fraction", 0.30))
    if abs((pair_fraction + val_fraction + test_fraction) - 1.0) > 1e-6:
        raise ValueError("pair_selection_fraction + validation_fraction + test_fraction must equal 1.0")

    purge = int(scfg.get("purge_horizon_days", 10))
    embargo = int(scfg.get("embargo_days", 2))
    n = len(index)
    train_cut = int(n * pair_fraction)
    val_cut = int(n * (pair_fraction + val_fraction))

    train_stop = max(1, train_cut - purge)
    val_start = min(n, train_cut + embargo)
    val_stop = max(val_start + 1, val_cut - purge)
    test_start = min(n - 1, val_cut + embargo)

    train_idx = index[:train_stop]
    val_idx = index[val_start:val_stop]
    test_idx = index[test_start:]
    if min(len(train_idx), len(val_idx), len(test_idx)) < 20:
        raise ValueError(
            f"Purged split is too small: train={len(train_idx)}, validation={len(val_idx)}, test={len(test_idx)}"
        )

    return TimeSplit(
        train_index=train_idx,
        validation_index=val_idx,
        test_index=test_idx,
        pair_selection_end=_safe_date(index, train_stop - 1),
        train_end=_safe_date(index, train_stop - 1),
        validation_start=_safe_date(index, val_start),
        validation_end=_safe_date(index, val_stop - 1),
        test_start=_safe_date(index, test_start),
        test_end=_safe_date(index, n - 1),
        purge_horizon_days=purge,
        embargo_days=embargo,
    )
