from __future__ import annotations

from copy import deepcopy

import pandas as pd

from .backtest import run_backtest, summarize_backtest
from .labels import build_convergence_labels
from .models import train_models
from .pairs import screen_pairs


def significance_sensitivity(training_prices: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    rows: list[dict] = []
    levels = cfg.get("m7", {}).get("significance_sensitivity_levels", [0.10, 0.05])
    for level in levels:
        local = deepcopy(cfg)
        local["pair_selection"]["max_eg_pvalue"] = float(level)
        local["pair_selection"]["max_residual_adf_pvalue"] = float(level)
        local["pair_selection"]["fdr_alpha"] = float(level)
        table, summary = screen_pairs(training_prices, local)
        selected = table.iloc[0]
        rows.append(
            {
                "significance_level": float(level),
                "pairs_tested": int(summary["pairs_tested"]),
                "pairs_passing_all": int(summary["pairs_passing_all"]),
                "selected_pair": f"{selected['asset_a']}-{selected['asset_b']}",
                "selected_passes_all": bool(selected["passes_all"]),
                "eg_pvalue": float(selected["eg_pvalue"]),
                "residual_adf_pvalue": float(selected["residual_adf_pvalue"]),
                "eg_fdr_pvalue": float(selected["eg_fdr_pvalue"]),
            }
        )
    return pd.DataFrame(rows)


def _scenario_configs(cfg: dict) -> list[tuple[str, dict]]:
    scenarios: list[tuple[str, dict]] = [("base", deepcopy(cfg))]

    def add(name: str, **updates) -> None:
        local = deepcopy(cfg)
        for dotted, value in updates.items():
            section, key = dotted.split(".", 1)
            local[section][key] = value
        scenarios.append((name, local))

    add("entry_z_1.25", **{"labels.entry_zscore": 1.25, "backtest.entry_z": 1.25})
    add("entry_z_1.75", **{"labels.entry_zscore": 1.75, "backtest.entry_z": 1.75})
    add(
        "lower_costs",
        **{
            "labels.transaction_cost_bps_per_leg": 1.0,
            "labels.slippage_bps_per_leg": 0.5,
            "backtest.transaction_cost_bps_per_leg": 1.0,
            "backtest.slippage_bps_per_leg": 0.5,
        },
    )
    add(
        "higher_costs",
        **{
            "labels.transaction_cost_bps_per_leg": 4.0,
            "labels.slippage_bps_per_leg": 2.0,
            "backtest.transaction_cost_bps_per_leg": 4.0,
            "backtest.slippage_bps_per_leg": 2.0,
        },
    )
    add("max_hold_10", **{"backtest.max_holding_days": 10})
    add("max_hold_40", **{"backtest.max_holding_days": 40})
    add("label_horizon_5", **{"labels.horizon_days": 5})
    return scenarios


def trading_rule_sensitivity(features: pd.DataFrame, split, cfg: dict, model_algorithms: list[str] | None = None) -> pd.DataFrame:
    """Retrain the ML decision layer for a compact, predeclared sensitivity set."""
    rows: list[dict] = []
    for offset, (scenario, local) in enumerate(_scenario_configs(cfg)):
        try:
            local["models"]["bootstrap_samples"] = 0
            if model_algorithms:
                local["models"]["algorithms"] = list(model_algorithms)
            labels = build_convergence_labels(features, local)
            bundles, metrics, predictions, selected_model, _ = train_models(features, labels, split, local)
            test_features = features.loc[features.index.intersection(split.test_index)]
            baseline_results, baseline_trades = run_backtest(test_features, local, "baseline")
            baseline = summarize_backtest(
                baseline_results, baseline_trades, "baseline", local, seed=500 + offset, compute_ci=False
            )
            accept = predictions[f"{selected_model}_accept"]
            ml_results, ml_trades = run_backtest(test_features, local, selected_model, accept_filter=accept)
            ml = summarize_backtest(ml_results, ml_trades, selected_model, local, seed=700 + offset, compute_ci=False)
            metric = metrics.loc[metrics["model"] == selected_model].iloc[0]
            rows.append(
                {
                    "scenario": scenario,
                    "selected_model": selected_model,
                    "test_auc": float(metric["test_auc"]),
                    "baseline_pnl": baseline["cumulative_net_pnl"],
                    "baseline_sharpe": baseline["sharpe"],
                    "ml_pnl": ml["cumulative_net_pnl"],
                    "ml_sharpe": ml["sharpe"],
                    "ml_max_drawdown": ml["max_drawdown"],
                    "ml_trades": ml["closed_trades"],
                    "ml_turnover": ml["turnover"],
                    "ml_total_cost": ml["total_cost"],
                    "ml_minus_baseline_pnl": ml["cumulative_net_pnl"] - baseline["cumulative_net_pnl"],
                }
            )
        except Exception as exc:
            rows.append({"scenario": scenario, "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def subperiod_stability(test_features: pd.DataFrame, cfg: dict, accept_filter: pd.Series, n_subperiods: int = 3) -> pd.DataFrame:
    rows: list[dict] = []
    boundaries = [round(i * len(test_features) / n_subperiods) for i in range(n_subperiods + 1)]
    for i in range(n_subperiods):
        sub = test_features.iloc[boundaries[i] : boundaries[i + 1]].copy()
        if len(sub) < 20:
            continue
        baseline_results, baseline_trades = run_backtest(sub, cfg, f"subperiod_{i+1}_baseline")
        baseline = summarize_backtest(
            baseline_results, baseline_trades, f"subperiod_{i+1}_baseline", cfg, seed=900 + i, compute_ci=False
        )
        accept = accept_filter.reindex(sub.index).fillna(0).astype(int)
        ml_results, ml_trades = run_backtest(sub, cfg, f"subperiod_{i+1}_ml", accept_filter=accept)
        ml = summarize_backtest(
            ml_results, ml_trades, f"subperiod_{i+1}_ml", cfg, seed=950 + i, compute_ci=False
        )
        rows.append(
            {
                "subperiod": i + 1,
                "start": str(sub.index.min().date()),
                "end": str(sub.index.max().date()),
                "baseline_pnl": baseline["cumulative_net_pnl"],
                "baseline_sharpe": baseline["sharpe"],
                "ml_pnl": ml["cumulative_net_pnl"],
                "ml_sharpe": ml["sharpe"],
                "ml_max_drawdown": ml["max_drawdown"],
                "ml_trades": ml["closed_trades"],
                "ml_minus_baseline_pnl": ml["cumulative_net_pnl"] - baseline["cumulative_net_pnl"],
            }
        )
    return pd.DataFrame(rows)
