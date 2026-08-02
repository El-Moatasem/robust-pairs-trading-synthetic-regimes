from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Loads and parses a YAML configuration file.

    Args:
        path (str | Path): Path to the YAML configuration file.

    Returns:
        dict[str, Any]: Parsed configuration mapping.

    Raises:
        ValueError: If the parsed YAML content is not a dictionary mapping.
    """
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration at {path} did not contain a YAML mapping.")
    return cfg


def set_seed(seed: int) -> None:
    """Sets global random seeds for pseudo-random number generators.

    Configures seeds for Python's built-in `random` module and `numpy.random`
    to ensure reproducible results.

    Args:
        seed (int): Seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)


def ensure_dirs(base: str | Path) -> dict[str, Path]:
    """Creates output directory structure if subdirectories do not already exist.

    Constructs paths for tables, figures, models, and logs relative to a base directory,
    creating each directory on the file system.

    Args:
        base (str | Path): Path to the base output directory.

    Returns:
        dict[str, Path]: A dictionary mapping directory keys ('base', 'tables', 'figures',
            'models', 'logs') to their resolved `Path` objects.
    """
    base = Path(base)
    result = {
        "base": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "models": base / "models",
        "logs": base / "logs",
    }
    for path in result.values():
        path.mkdir(parents=True, exist_ok=True)
    return result


def save_json(value: Any, path: str | Path) -> None:
    """Serializes and saves a data object to a JSON file.

    Creates parent directories if necessary and formats the output with an indentation
    level of 2.

    Args:
        value (Any): Python object or data structure to serialize.
        path (str | Path): Target output file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, default=str)


def as_builtin(value: Any) -> Any:
    """Convert numpy/pandas scalar values to JSON-safe Python values.

    Recursively traverses dictionaries, lists, and tuples, converting any `np.generic`
    scalars (e.g., `np.int64`, `np.float64`) to standard Python types via `.item()`.

    Args:
        value (Any): Input object, collection, or scalar value.

    Returns:
        Any: Equivalent Python built-in representation suitable for JSON serialization.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): as_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_builtin(v) for v in value]
    return value
