from __future__ import annotations

import numpy as np
import pandas as pd


def flatten_model_metrics(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, payload in results.items():
        if not isinstance(payload, dict) or "metrics" not in payload:
            continue
        row = {"model": model_name}
        for k, v in payload["metrics"].items():
            if k != "confusion_matrix":
                row[k] = v
        rows.append(row)
    return pd.DataFrame(rows)


def compare_strategy_metrics(metrics: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame([{"strategy": k, **v} for k, v in metrics.items()])


def robustness_table(scenario_metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for scenario, m in scenario_metrics.items():
        row = {"scenario": scenario}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)
