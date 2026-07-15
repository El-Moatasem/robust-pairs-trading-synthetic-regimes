from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.backtest import backtest_zscore, summarize_backtest
from src.data import load_public_or_synthetic_data
from src.evaluation import make_initial_inferences
from src.features import build_feature_frame
from src.labels import build_convergence_labels
from src.models import train_models
from src.pairs import screen_pairs, select_top_pair
from src.synthetic import generate_synthetic_spreads, summarize_scenarios
from src.utils import ensure_dirs, load_config, save_json, set_seed
from src.visualization import plot_equity_curve, plot_scenario_volatility, plot_feature_importance


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ML-enhanced pairs-trading capstone pipeline.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"].get("random_seed", 42)))
    dirs = ensure_dirs(cfg["outputs"]["dir"])

    prices, data_summary = load_public_or_synthetic_data(cfg)
    prices.to_csv(dirs["tables"] / "prices_sample.csv")
    save_json(data_summary.__dict__, dirs["tables"] / "data_summary.json")

    pair_table = screen_pairs(
        prices,
        min_abs_corr=float(cfg["pair_selection"]["min_abs_corr"]),
        max_pvalue=float(cfg["pair_selection"]["max_cointegration_pvalue"]),
        top_n=int(cfg["pair_selection"]["top_n"]),
    )
    pair_table.to_csv(dirs["tables"] / "candidate_pairs.csv", index=False)
    asset_a, asset_b, hedge_ratio = select_top_pair(pair_table)

    features = build_feature_frame(prices, asset_a, asset_b, cfg, hedge_ratio)
    features.to_csv(dirs["tables"] / "feature_frame.csv")
    labels = build_convergence_labels(features, cfg)
    labels.to_csv(dirs["tables"] / "labels.csv")

    bundles, model_metrics, predictions = train_models(features, labels, cfg)
    model_metrics.to_csv(dirs["tables"] / "model_metrics.csv", index=False)
    predictions.to_csv(dirs["tables"] / "model_predictions.csv")

    baseline = backtest_zscore(features, cfg)
    baseline.to_csv(dirs["tables"] / "backtest_baseline.csv")

    # Use the best F1 model for ML-filtered backtest.
    best_model_name = model_metrics.sort_values("f1", ascending=False).iloc[0]["model"]
    accept_col = f"{best_model_name}_accept"
    accept_filter = predictions[accept_col]
    ml_filtered = backtest_zscore(features, cfg, accept_filter=accept_filter)
    ml_filtered.to_csv(dirs["tables"] / "backtest_ml_filtered.csv")

    strategy_metrics = pd.DataFrame([
        summarize_backtest(baseline, "Baseline z-score"),
        summarize_backtest(ml_filtered, f"ML-filtered ({best_model_name})"),
    ])
    strategy_metrics.to_csv(dirs["tables"] / "strategy_metrics.csv", index=False)

    scenarios = generate_synthetic_spreads(features["spread"], cfg)
    scenarios.to_csv(dirs["tables"] / "synthetic_scenarios.csv")
    scenario_summary = summarize_scenarios(scenarios)
    scenario_summary.to_csv(dirs["tables"] / "synthetic_scenario_summary.csv", index=False)

    plot_equity_curve(baseline, dirs["figures"] / "baseline_equity_curve.png", "Baseline Pairs-Trading Equity Curve")
    plot_equity_curve(ml_filtered, dirs["figures"] / "ml_filtered_equity_curve.png", "ML-Filtered Pairs-Trading Equity Curve")
    plot_scenario_volatility(scenario_summary, dirs["figures"] / "synthetic_scenario_volatility.png")
    plot_feature_importance(bundles[best_model_name].feature_importance, dirs["figures"] / "feature_importance.png", f"Feature Importance: {best_model_name}")

    with open(dirs["base"] / "initial_findings.md", "w", encoding="utf-8") as f:
        f.write(make_initial_inferences(strategy_metrics, model_metrics, scenario_summary))

    print("Pipeline completed successfully.")
    print(f"Data source: {data_summary.source}")
    print(f"Selected pair: {asset_a}-{asset_b}, hedge ratio={hedge_ratio:.4f}")
    print(f"Best model: {best_model_name}")
    print(f"Outputs written to: {dirs['base'].resolve()}")


if __name__ == "__main__":
    main()
