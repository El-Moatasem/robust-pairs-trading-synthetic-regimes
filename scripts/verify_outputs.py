from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_FILES = [
    "run_summary.json",
    "initial_findings.md",
    "config_used.yaml",
    "tables/data_summary.json",
    "tables/time_split_summary.json",
    "tables/candidate_pairs.csv",
    "tables/feature_definitions.csv",
    "tables/labels.csv",
    "tables/model_metrics.csv",
    "tables/strategy_metrics_test.csv",
    "tables/synthetic_calibration.json",
    "tables/synthetic_regime_summary.csv",
    "figures/pair_spread_and_signal.png",
    "figures/out_of_sample_equity_comparison.png",
    "figures/out_of_sample_roc_curves.png",
    "figures/selected_model_feature_importance.png",
    "figures/synthetic_regime_performance.png",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify required pipeline outputs and research controls.")
    parser.add_argument("--outputs", default="outputs")
    args = parser.parse_args()
    root = Path(args.outputs)

    missing = [relative for relative in REQUIRED_FILES if not (root / relative).exists()]
    if missing:
        raise SystemExit("Missing required output files:\n- " + "\n- ".join(missing))

    with (root / "tables" / "data_summary.json").open(encoding="utf-8") as handle:
        data_summary = json.load(handle)
    with (root / "tables" / "time_split_summary.json").open(encoding="utf-8") as handle:
        split = json.load(handle)
    candidates = pd.read_csv(root / "tables" / "candidate_pairs.csv")
    models = pd.read_csv(root / "tables" / "model_metrics.csv")
    strategies = pd.read_csv(root / "tables" / "strategy_metrics_test.csv")

    required_candidate_columns = {
        "eg_pvalue",
        "residual_adf_pvalue",
        "eg_fdr_pvalue",
        "passes_i1",
        "passes_all",
    }
    if not required_candidate_columns.issubset(candidates.columns):
        raise SystemExit("Candidate-pair output is missing one or more conservative diagnostic columns.")
    if split.get("purge_horizon_days", 0) <= 0 or split.get("embargo_days", -1) < 0:
        raise SystemExit("Purge/embargo settings were not recorded correctly.")
    if not any(column.startswith("validation_") for column in models.columns):
        raise SystemExit("Validation metrics are missing from model output.")
    if not any(column.startswith("test_") for column in models.columns):
        raise SystemExit("Test metrics are missing from model output.")
    if len(strategies) < 2 or "Baseline z-score" not in set(strategies["strategy"]):
        raise SystemExit("Baseline and ML strategy comparison was not produced.")
    if data_summary.get("is_synthetic") is None:
        raise SystemExit("The output does not explicitly identify whether data are synthetic.")

    print("Output verification passed.")
    print(f"Data source: {data_summary['source']}")
    print(f"Synthetic data: {data_summary['is_synthetic']}")
    print(f"Candidate pairs displayed: {len(candidates)}")
    print(f"Models evaluated: {len(models)}")
    print(f"Strategies evaluated: {len(strategies)}")


if __name__ == "__main__":
    main()
