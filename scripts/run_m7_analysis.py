from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest import run_backtest
from src.data import load_prices
from src.eda import generate_eda_figures
from src.features import build_feature_frame
from src.independent_backtest import compare_backtest_engines, replay_trade_backtest
from src.labels import build_convergence_labels
from src.m7_visualization import (
    plot_subperiod_stability,
    plot_trading_rule_sensitivity,
    plot_walk_forward_diagnostics,
    plot_walk_forward_pnl,
)
from src.models import train_models
from src.pairs import screen_pairs, select_top_pair
from src.sensitivity import significance_sensitivity, subperiod_stability, trading_rule_sensitivity
from src.splits import make_time_split
from src.utils import ensure_dirs, load_config, save_json, set_seed
from src.walkforward import run_walk_forward, run_prespecified_pair_walk_forward


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Module 7/final robustness extensions.")
    parser.add_argument("--config", default="config/config_m7_public.yaml")
    parser.add_argument("--require-public-data", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"].get("random_seed", 42))
    set_seed(seed)
    dirs = ensure_dirs(cfg["outputs"]["dir"])
    m7_dir = Path(dirs["base"]) / "m7"
    tables = m7_dir / "tables"
    figures = m7_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    prices, data_summary = load_prices(cfg, require_public=args.require_public_data)
    split = make_time_split(prices.index, cfg)
    training_prices = prices.loc[prices.index.intersection(split.train_index)]
    candidate_pairs, pair_summary = screen_pairs(training_prices, cfg)
    asset_a, asset_b, alpha, beta, pair_status = select_top_pair(candidate_pairs)
    selected_pair = candidate_pairs.iloc[0]
    features = build_feature_frame(prices, asset_a, asset_b, cfg, alpha, beta)
    labels = build_convergence_labels(features, cfg)
    bundles, model_metrics, predictions, selected_model, supervised_summary = train_models(features, labels, split, cfg)
    test_features = features.loc[features.index.intersection(split.test_index)]

    # EDA appendix figures requested by instructor feedback.
    eda_paths = generate_eda_figures(prices, features, figures)

    # Independent trade-level replay validation of both baseline and validation-selected ML strategy.
    primary_base, primary_base_trades = run_backtest(test_features, cfg, "baseline")
    independent_base_trades, independent_base_summary = replay_trade_backtest(test_features, cfg)
    base_compare = compare_backtest_engines(primary_base_trades, independent_base_trades)
    base_compare["strategy"] = "baseline"

    selected_accept = predictions[f"{selected_model}_accept"]
    primary_ml, primary_ml_trades = run_backtest(test_features, cfg, selected_model, selected_accept)
    independent_ml_trades, independent_ml_summary = replay_trade_backtest(test_features, cfg, selected_accept)
    ml_compare = compare_backtest_engines(primary_ml_trades, independent_ml_trades)
    ml_compare["strategy"] = selected_model
    pd.DataFrame([base_compare, ml_compare]).to_csv(tables / "independent_backtest_comparison.csv", index=False)
    independent_base_trades.to_csv(tables / "independent_baseline_trades.csv", index=False)
    independent_ml_trades.to_csv(tables / "independent_selected_model_trades.csv", index=False)

    # Statistical and trading-rule sensitivity.
    sig = significance_sensitivity(training_prices, cfg)
    sig.to_csv(tables / "significance_sensitivity.csv", index=False)
    trading = trading_rule_sensitivity(features, split, cfg, model_algorithms=[selected_model])
    trading.to_csv(tables / "trading_rule_sensitivity.csv", index=False)

    # Subperiod stability within the untouched fixed public test window.
    n_subperiods = int(cfg.get("m7", {}).get("subperiods", 3))
    subperiod = subperiod_stability(test_features, cfg, selected_accept, n_subperiods=n_subperiods)
    subperiod.to_csv(tables / "subperiod_stability.csv", index=False)

    # Expanding-window walk-forward analysis; pair selection and model selection repeat in every fold.
    walk_forward, walk_pairs = run_walk_forward(prices, cfg)
    walk_forward.to_csv(tables / "walk_forward_summary.csv", index=False)
    walk_pairs.to_csv(tables / "walk_forward_pair_screens.csv", index=False)
    profitable = walk_forward.loc[walk_forward.get("passed_and_profitable", False) == True].copy()
    profitable.to_csv(tables / "passed_and_profitable_walk_forward.csv", index=False)
    prespecified = run_prespecified_pair_walk_forward(prices, cfg, pair_screens=walk_pairs)
    prespecified.to_csv(tables / "prespecified_pair_walk_forward.csv", index=False)
    prespecified_profitable = prespecified.loc[prespecified.get("profitable_test", False) == True].copy() if len(prespecified) else pd.DataFrame()
    prespecified_profitable.to_csv(tables / "prespecified_pairs_passed_and_profitable.csv", index=False)

    # Report-ready M7 visuals: walk-forward performance/diagnostics and sensitivity stability.
    m7_visual_paths = [
        plot_walk_forward_pnl(walk_forward, figures),
        plot_walk_forward_diagnostics(walk_forward, figures),
        plot_trading_rule_sensitivity(trading, figures),
        plot_subperiod_stability(subperiod, figures),
    ]

    summary = {
        "data": data_summary.to_dict(),
        "fixed_split_selected_pair": f"{asset_a}-{asset_b}",
        "fixed_split_pair_status": pair_status,
        "fixed_split_pairs_passing_all": int(pair_summary["pairs_passing_all"]),
        "fixed_split_selected_model": selected_model,
        "walk_forward_folds": int(len(walk_forward)),
        "walk_forward_folds_with_full_pair_pass": int((walk_forward.get("pair_status") == "passed_all").sum()) if len(walk_forward) else 0,
        "walk_forward_passed_and_profitable_folds": int(walk_forward.get("passed_and_profitable", pd.Series(dtype=bool)).sum()) if len(walk_forward) else 0,
        "prespecified_passed_and_profitable_cases": int(len(prespecified_profitable)),
        "independent_backtest_baseline_match": bool(base_compare["trade_count_match"] and base_compare["pnl_match_within_1e_10"]),
        "independent_backtest_selected_model_match": bool(ml_compare["trade_count_match"] and ml_compare["pnl_match_within_1e_10"]),
        "eda_figures": [str(path) for path in eda_paths],
        "m7_visuals": [str(path) for path in m7_visual_paths],
    }
    save_json(summary, m7_dir / "m7_summary.json")

    print("M7 analysis completed successfully.")
    print(f"Data source: {data_summary.source}")
    print(f"Fixed split pair: {asset_a}-{asset_b} ({pair_status})")
    print(f"Walk-forward folds: {len(walk_forward)}")
    if len(profitable):
        print("Walk-forward discovery-universe pairs that passed all screens and produced positive selected-ML test PnL:")
        print(profitable[["fold", "selected_pair", "test_start", "test_end", "ml_pnl", "ml_sharpe"]].to_string(index=False))
    else:
        print("No discovery-universe walk-forward selected pair both passed all conservative screens and produced positive selected-ML test PnL.")
    if len(prespecified_profitable):
        print("Pre-specified-pair sensitivity cases that passed diagnostics and had positive OOS PnL:")
        print(prespecified_profitable[["fold", "pair", "test_start", "test_end", "ml_pnl", "ml_sharpe"]].to_string(index=False))
    print(f"Outputs: {m7_dir.resolve()}")


if __name__ == "__main__":
    main()
