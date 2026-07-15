from __future__ import annotations

import pandas as pd


def make_initial_inferences(strategy_metrics: pd.DataFrame, model_metrics: pd.DataFrame, scenario_summary: pd.DataFrame) -> str:
    best_strategy = strategy_metrics.sort_values("sharpe", ascending=False).iloc[0]
    best_model = model_metrics.sort_values("f1", ascending=False).iloc[0]
    stress = scenario_summary.sort_values("std", ascending=False).head(3)
    text = []
    text.append("# Initial Findings and Temporary Results\n")
    text.append("The current codebase was run on reproducible daily sample data so that the pipeline can be reviewed without internet access. These results are temporary and intended to demonstrate code development, not final investment performance.\n")
    text.append(f"- Best temporary strategy by Sharpe: {best_strategy['strategy']} with Sharpe {best_strategy['sharpe']:.3f}.\n")
    text.append(f"- Best temporary model by F1: {best_model['model']} with F1 {best_model['f1']:.3f}.\n")
    text.append("- Synthetic scenarios with the largest spread volatility are: " + ", ".join(stress['scenario'].tolist()) + ".\n")
    text.append("- The initial results suggest the project can already select candidate pairs, construct spread features, produce accept/reject labels, train ML filters, run baseline/ML-filtered backtests, and generate robustness scenarios.\n")
    return "\n".join(text)
