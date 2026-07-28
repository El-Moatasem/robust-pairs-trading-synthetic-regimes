from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import joblib
import pandas as pd

from src.backtest import run_backtest, summarize_backtest
from src.data import load_prices
from src.evaluation import make_initial_findings
from src.features import build_feature_frame
from src.labels import build_convergence_labels
from src.models import train_models
from src.pairs import screen_pairs, select_top_pair
from src.splits import make_time_split
from src.synthetic import evaluate_synthetic_regimes, summarize_regimes
from src.utils import as_builtin, ensure_dirs, load_config, save_json, set_seed
from src.visualization import (
    plot_equity_comparison,
    plot_feature_importance,
    plot_pair_diagnostics,
    plot_roc_curves,
    plot_scenario_spreads,
    plot_synthetic_regime_performance,
)


def _feature_definitions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("signal_zscore", "Lagged rolling z-score of the fixed-hedge-ratio log spread."),
            ("abs_zscore", "Absolute value of the lagged spread z-score."),
            ("spread_change_lagged", "One-period lagged change in the log spread."),
            ("spread_volatility", "Lagged rolling standard deviation of spread changes."),
            ("rolling_corr", "Lagged rolling correlation of the two assets' log returns."),
            ("spread_drawdown", "Lagged spread minus its rolling maximum."),
            ("half_life", "Lagged rolling AR(1)/OU half-life estimate in trading days."),
            ("deviation_persistence", "Lagged count of consecutive observations with |z| above the configured threshold."),
            ("regime_stress_proxy", "Equal-weighted rolling percentile of spread volatility and weakening correlation."),
        ],
        columns=["feature", "definition"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the revised robust pairs-trading research pipeline.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to the YAML configuration.")
    parser.add_argument(
        "--require-public-data",
        action="store_true",
        help="Fail rather than fall back to synthetic data if public data cannot be loaded.",
    )
    parser.add_argument("--output-dir", default=None, help="Override outputs.dir from the configuration.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.output_dir:
        cfg["outputs"]["dir"] = args.output_dir
    seed = int(cfg["project"].get("random_seed", 42))
    set_seed(seed)
    dirs = ensure_dirs(cfg["outputs"]["dir"])
    shutil.copy2(args.config, dirs["base"] / "config_used.yaml")

    prices, data_summary = load_prices(cfg, require_public=args.require_public_data)
    prices.to_csv(dirs["tables"] / "prices.csv")
    save_json(data_summary.to_dict(), dirs["tables"] / "data_summary.json")

    split = make_time_split(prices.index, cfg)
    split_summary = split.summary()
    save_json(split_summary, dirs["tables"] / "time_split_summary.json")

    pair_training_prices = prices.loc[prices.index.intersection(split.train_index)]
    candidate_pairs, pair_summary = screen_pairs(pair_training_prices, cfg)
    candidate_pairs.to_csv(dirs["tables"] / "candidate_pairs.csv", index=False)
    save_json(pair_summary, dirs["tables"] / "pair_selection_summary.json")
    asset_a, asset_b, alpha, hedge_ratio, pair_status = select_top_pair(candidate_pairs)
    selected_pair = candidate_pairs.iloc[0].copy()

    features = build_feature_frame(prices, asset_a, asset_b, cfg, alpha, hedge_ratio)
    features.to_csv(dirs["tables"] / "feature_frame.csv")
    _feature_definitions().to_csv(dirs["tables"] / "feature_definitions.csv", index=False)

    labels = build_convergence_labels(features, cfg)
    labels.to_csv(dirs["tables"] / "labels.csv")
    label_summary = pd.DataFrame(
        [
            {
                "signals_labeled": len(labels),
                "accepted_signals": int(labels["accept_signal"].sum()),
                "acceptance_rate": float(labels["accept_signal"].mean()) if len(labels) else 0.0,
                "median_holding_days": float(labels["realized_holding_days"].median()) if len(labels) else 0.0,
                "median_net_return": float(labels["realized_net_return"].median()) if len(labels) else 0.0,
            }
        ]
    )
    label_summary.to_csv(dirs["tables"] / "label_summary.csv", index=False)

    bundles, model_metrics, predictions, selected_model, supervised_summary = train_models(
        features, labels, split, cfg
    )
    model_metrics.to_csv(dirs["tables"] / "model_metrics.csv", index=False)
    predictions.to_csv(dirs["tables"] / "model_predictions_test.csv")
    split_summary.update(supervised_summary)
    save_json(split_summary, dirs["tables"] / "time_split_summary.json")
    for name, bundle in bundles.items():
        joblib.dump(bundle.model, dirs["models"] / f"{name}.joblib")
        bundle.feature_importance.to_csv(dirs["tables"] / f"feature_importance_{name}.csv", index=False)

    test_features = features.loc[features.index.intersection(split.test_index)].copy()
    results_by_name: dict[str, pd.DataFrame] = {}
    strategy_rows: list[dict] = []

    baseline_results, baseline_trades = run_backtest(test_features, cfg, "Baseline z-score")
    baseline_results.to_csv(dirs["tables"] / "backtest_baseline_test.csv")
    baseline_trades.to_csv(dirs["tables"] / "trades_baseline_test.csv", index=False)
    results_by_name["Baseline z-score"] = baseline_results
    strategy_rows.append(summarize_backtest(baseline_results, baseline_trades, "Baseline z-score", cfg, seed))

    for offset, name in enumerate(bundles):
        accept = predictions[f"{name}_accept"]
        model_results, model_trades = run_backtest(test_features, cfg, f"ML-filtered ({name})", accept)
        model_results.to_csv(dirs["tables"] / f"backtest_{name}_test.csv")
        model_trades.to_csv(dirs["tables"] / f"trades_{name}_test.csv", index=False)
        results_by_name[f"ML-filtered ({name})"] = model_results
        strategy_rows.append(
            summarize_backtest(model_results, model_trades, f"ML-filtered ({name})", cfg, seed + offset + 1)
        )

    strategy_metrics = pd.DataFrame(strategy_rows)
    strategy_metrics.to_csv(dirs["tables"] / "strategy_metrics_test.csv", index=False)

    training_features = features.loc[features.index.intersection(split.train_index)]
    scenario_table, sample_paths, calibration = evaluate_synthetic_regimes(
        training_features=training_features,
        cfg=cfg,
        asset_a=asset_a,
        asset_b=asset_b,
        alpha=alpha,
        beta=hedge_ratio,
        selected_bundle=bundles[selected_model],
    )
    scenario_table.to_csv(dirs["tables"] / "synthetic_scenario_metrics.csv", index=False)
    sample_paths.to_csv(dirs["tables"] / "synthetic_sample_paths.csv", index=False)
    regime_summary = summarize_regimes(scenario_table)
    regime_summary.to_csv(dirs["tables"] / "synthetic_regime_summary.csv", index=False)
    save_json(calibration.to_dict(), dirs["tables"] / "synthetic_calibration.json")

    plot_pair_diagnostics(features, dirs["figures"] / "pair_spread_and_signal.png", asset_a, asset_b)
    plot_equity_comparison(results_by_name, dirs["figures"] / "out_of_sample_equity_comparison.png")
    plot_feature_importance(
        bundles[selected_model].feature_importance,
        dirs["figures"] / "selected_model_feature_importance.png",
        f"Selected Model Feature Importance: {selected_model}",
    )
    plot_roc_curves(predictions, list(bundles), dirs["figures"] / "out_of_sample_roc_curves.png")
    if not scenario_table.empty:
        plot_synthetic_regime_performance(
            scenario_table, dirs["figures"] / "synthetic_regime_performance.png"
        )
    if not sample_paths.empty:
        plot_scenario_spreads(sample_paths, dirs["figures"] / "synthetic_sample_spreads.png")

    run_summary = {
        "project": cfg["project"],
        "data": data_summary.to_dict(),
        "time_split": split_summary,
        "pair_selection": pair_summary,
        "selected_pair": as_builtin(selected_pair.to_dict()),
        "pair_status": pair_status,
        "selected_model": selected_model,
        "synthetic_calibration": calibration.to_dict(),
        "important_caveat": (
            "Synthetic-mode results demonstrate the corrected methodology and software workflow; "
            "they are not evidence of historical investment profitability."
            if data_summary.is_synthetic
            else "Results use public adjusted-close data and remain subject to the documented model and execution assumptions."
        ),
    }
    save_json(run_summary, dirs["base"] / "run_summary.json")

    findings = make_initial_findings(
        data_summary=data_summary.to_dict(),
        pair_summary=pair_summary,
        selected_pair=selected_pair,
        split_summary=split_summary,
        model_metrics=model_metrics,
        strategy_metrics=strategy_metrics,
        regime_summary=regime_summary,
    )
    (dirs["base"] / "initial_findings.md").write_text(findings, encoding="utf-8")

    print("Pipeline completed successfully.")
    print(f"Data source: {data_summary.source}")
    if data_summary.fallback_reason:
        print(f"Public-data fallback reason: {data_summary.fallback_reason}")
    print(f"Pair-selection window ends: {split.pair_selection_end}")
    print(
        f"Selected pair: {asset_a}-{asset_b}; alpha={alpha:.4f}; hedge ratio={hedge_ratio:.4f}; "
        f"status={pair_status}"
    )
    print(f"Selected model (validation only): {selected_model}")
    print(f"Test window: {split.test_start} to {split.test_end}")
    print(f"Outputs written to: {dirs['base'].resolve()}")


if __name__ == "__main__":
    main()
