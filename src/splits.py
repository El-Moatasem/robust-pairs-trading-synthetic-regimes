from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    """Immutable data container holding chronological train/validation/test indices and date metadata.

    Attributes:
        train_index (pd.Index): Index subset corresponding to the training split.
        validation_index (pd.Index): Index subset corresponding to the validation split.
        test_index (pd.Index): Index subset corresponding to the testing split.
        pair_selection_end (str): End date string for the pair selection/train window.
        train_end (str): End date string for the training split.
        validation_start (str): Start date string for the validation split.
        validation_end (str): End date string for the validation split.
        test_start (str): Start date string for the test split.
        test_end (str): End date string for the test split.
        purge_horizon_days (int): Number of days purged before partition boundaries to prevent label leakage.
        embargo_days (int): Number of days embargoed after partition boundaries to prevent post-boundary leakage.
    """

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
        """Converts the TimeSplit instance into a summary dictionary with split lengths.

        Returns:
            dict: Dictionary representation of time split metadata where index fields
                are replaced with their respective integer observation counts.
        """
        result = asdict(self)
        result["train_index"] = len(self.train_index)
        result["validation_index"] = len(self.validation_index)
        result["test_index"] = len(self.test_index)
        return result


def _safe_date(index: pd.Index, position: int) -> str:
    """Safely extracts an ISO-formatted date string from a pandas Index at a given position.

    Args:
        index (pd.Index): Pandas index containing date or timestamp objects.
        position (int): Integer index position of the element to extract.

    Returns:
        str: Date string formatted as 'YYYY-MM-DD' if index contains date-like objects,
            otherwise the string representation of the value.
    """
    value = index[position]
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)


def make_time_split(index: pd.Index, cfg: dict) -> TimeSplit:
    """Create chronological train/validation/test sets with purge and embargo gaps.

    Labels use future returns up to `purge_horizon_days`, so observations immediately before a
    boundary are removed from the earlier set. An additional embargo removes observations after
    each boundary from the next set. This prevents overlapping label horizons across partitions.

    Args:
        index (pd.Index): Sequential time-series pandas index (e.g., trading dates).
        cfg (dict): Configuration dictionary containing split settings under the `"split"` key
            (e.g., `pair_selection_fraction`, `validation_fraction`, `test_fraction`,
            `purge_horizon_days`, `embargo_days`).

    Returns:
        TimeSplit: A populated `TimeSplit` dataclass containing the purged index partitions
            and string date boundary attributes.

    Raises:
        ValueError: If total observations are fewer than 150, if partition fractions
            do not sum to 1.0, or if any resulting purged partition has fewer than 20 observations.
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
