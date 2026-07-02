from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import pandas as pd

from src.data import DataConfig, get_price_data
from src.pairs import screen_candidate_pairs
from src.features import construct_pair_features
from src.backtest import generate_baseline_signals, backtest_pair_strategy, apply_ml_filter
from src.labels import build_convergence_labels, get_model_dataset
from src.models import train_trade_filters
from src.evaluation import flatten_model_metrics, compare_strategy_metrics
from src.synthetic import generate_synthetic_scenarios, scenario_summary
from src.reporting import ensure_output_dirs, save_table, write_initial_findings_markdown
from src.visualization import plot_equity_curve, plot_scenario_distribution


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    tables_dir, figures_dir = ensure_output_dirs(cfg["outputs"]["tables_dir"], cfg["outputs"]["figures_dir"])

    data_cfg = DataConfig(
        tickers=cfg["data"]["tickers"],
        start_date=cfg["data"].get("start_date", "2018-01-01"),
        end_date=cfg["data"].get("end_date", "2025-12-31"),
        price_field=cfg["data"].get("price_field", "Adj Close"),
        use_sample_data=cfg["data"].get("use_sample_data", True),
        random_state=cfg["ml"].get("random_state", 42),
    )
    prices = get_price_data(data_cfg)
    prices.to_csv(tables_dir / "price_panel.csv")

    pairs = screen_candidate_pairs(
        prices,
        min_abs_correlation=cfg["pair_selection"]["min_abs_correlation"],
        max_cointegration_pvalue=cfg["pair_selection"]["max_cointegration_pvalue"],
        top_n=cfg["pair_selection"]["top_n_pairs"],
    )
    save_table(pairs, tables_dir / "candidate_pairs.csv")
    if pairs.empty:
        raise RuntimeError("No candidate pairs found. Lower thresholds or change universe.")

    first_pair = pairs.iloc[0]
    features = construct_pair_features(
        prices,
        first_pair["asset_a"],
        first_pair["asset_b"],
        zscore_window=cfg["features"]["zscore_window"],
        volatility_window=cfg["features"]["volatility_window"],
        correlation_window=cfg["features"]["correlation_window"],
    )
    features.to_csv(tables_dir / "features_first_pair.csv")

    signals = generate_baseline_signals(
        features,
        entry_z=cfg["strategy"]["entry_z"],
        exit_z=cfg["strategy"]["exit_z"],
        stop_z=cfg["strategy"]["stop_z"],
    )
    baseline_bt, baseline_metrics = backtest_pair_strategy(signals, cfg["strategy"]["transaction_cost_bps"])
    baseline_bt.to_csv(tables_dir / "baseline_backtest.csv")
    plot_equity_curve(baseline_bt, figures_dir / "baseline_equity_curve.png", "Baseline Pairs-Trading Equity Curve")

    labeled = build_convergence_labels(signals, exit_z=cfg["strategy"]["exit_z"], max_holding_days=cfg["strategy"]["max_holding_days"])
    X, y = get_model_dataset(labeled)
    model_results = train_trade_filters(
        X,
        y,
        model_names=cfg["ml"]["models"],
        test_size=cfg["ml"]["test_size"],
        random_state=cfg["ml"]["random_state"],
    )
    model_metrics = flatten_model_metrics(model_results)
    save_table(model_metrics, tables_dir / "model_metrics.csv")

    strategy_metrics = {"baseline": baseline_metrics}
    # Run one ML-filtered strategy if model probabilities are available.
    for model_name, payload in model_results.items():
        if isinstance(payload, dict) and "probabilities" in payload:
            filtered = apply_ml_filter(signals, payload["probabilities"], threshold=cfg["ml"]["probability_threshold"])
            filtered_bt, filtered_metrics = backtest_pair_strategy(filtered, cfg["strategy"]["transaction_cost_bps"])
            strategy_metrics[f"ml_filtered_{model_name}"] = filtered_metrics
            filtered_bt.to_csv(tables_dir / f"ml_filtered_{model_name}_backtest.csv")
            break
    strategy_metrics_df = compare_strategy_metrics(strategy_metrics)
    save_table(strategy_metrics_df, tables_dir / "strategy_metrics.csv")

    scenarios = generate_synthetic_scenarios(
        n_scenarios=cfg["synthetic"]["n_scenarios"],
        n_steps=cfg["synthetic"]["n_steps"],
        base_mean_reversion=cfg["synthetic"]["base_mean_reversion"],
        base_volatility=cfg["synthetic"]["base_volatility"],
        jump_probability=cfg["synthetic"]["jump_probability"],
        jump_scale=cfg["synthetic"]["jump_scale"],
        random_state=cfg["ml"]["random_state"],
    )
    scen_summary = scenario_summary(scenarios)
    save_table(scen_summary, tables_dir / "synthetic_scenario_summary.csv")
    plot_scenario_distribution(scen_summary, figures_dir / "synthetic_scenario_volatility.png")

    write_initial_findings_markdown(base_path := Path("outputs") / "initial_findings.md", pairs, model_metrics, strategy_metrics_df)
    print(f"Pipeline complete. Tables saved to {tables_dir}; figures saved to {figures_dir}; initial findings saved to {base_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MScFE 690 capstone research pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config")
    args = parser.parse_args()
    main(args.config)
