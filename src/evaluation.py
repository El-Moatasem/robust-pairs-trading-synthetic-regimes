from __future__ import annotations

import pandas as pd


def make_initial_findings(
    data_summary: dict,
    pair_summary: dict,
    selected_pair: pd.Series,
    split_summary: dict,
    model_metrics: pd.DataFrame,
    strategy_metrics: pd.DataFrame,
    regime_summary: pd.DataFrame,
) -> str:
    selected_model_row = model_metrics.loc[
        model_metrics["validation_f1"].idxmax()
    ] if "validation_f1" in model_metrics else model_metrics.iloc[0]
    best_strategy = strategy_metrics.sort_values("sharpe", ascending=False).iloc[0]
    lines = [
        "# Revised Preliminary Findings",
        "",
        "These outputs are research-development results. They do not constitute investment advice or final evidence of profitability.",
        "",
        "## Data and leakage controls",
        f"- Data source: `{data_summary['source']}`.",
        f"- Synthetic data used: `{data_summary['is_synthetic']}`.",
        f"- Pair selection was performed only through {split_summary['pair_selection_end']}.",
        f"- Purge horizon: {split_summary['purge_horizon_days']} days; embargo: {split_summary['embargo_days']} days.",
        "",
        "## Pair selection",
        f"- Pairs tested: {pair_summary['pairs_tested']}; pairs passing all conservative filters: {pair_summary['pairs_passing_all']}.",
        f"- Selected pair: {selected_pair['asset_a']}-{selected_pair['asset_b']} ({pair_summary['selection_status']}).",
        f"- Engle-Granger p-value: {selected_pair['eg_pvalue']:.4f}; residual ADF p-value: {selected_pair['residual_adf_pvalue']:.4f}; FDR-adjusted p-value: {selected_pair['eg_fdr_pvalue']:.4f}.",
        "",
        "## ML validation and testing",
        f"- Model selection used validation data only: {split_summary['model_selection_basis']}.",
        f"- Selected model: {split_summary['selected_model']} with validation F1 {selected_model_row['validation_f1']:.3f} and test F1 {selected_model_row['test_f1']:.3f}.",
        f"- Test positive-class prevalence: {split_summary['test_prevalence']:.3f}.",
        "",
        "## Out-of-sample trading",
        f"- Best reported test-period strategy by Sharpe: {best_strategy['strategy']} with Sharpe {best_strategy['sharpe']:.3f}.",
        "- Classifier quality and trading profitability are reported separately; predictive metrics do not imply profitable execution.",
        "",
        "## Synthetic robustness",
    ]
    if regime_summary.empty:
        lines.append("- Synthetic scenario evaluation did not produce enough observations.")
    else:
        for _, row in regime_summary.iterrows():
            lines.append(
                f"- {row['regime']}: ML outperformed baseline in {row['ml_outperforms_fraction']:.1%} of scenarios; "
                f"mean PnL difference {row['mean_ml_minus_baseline_pnl']:.4f}."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "The revised pipeline explicitly distinguishes synthetic from public data, freezes pair-selection and hedge-ratio estimates before the test period, purges overlapping label horizons, selects models on validation data, and reports test-period trading results with uncertainty intervals.",
        ]
    )
    return "\n".join(lines) + "\n"
