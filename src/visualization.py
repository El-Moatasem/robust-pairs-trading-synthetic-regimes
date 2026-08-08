from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay


def plot_pair_diagnostics(features: pd.DataFrame, path: str | Path, asset_a: str, asset_b: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axes[0].plot(features.index, features["spread"])
    axes[0].set_title(f"Training-Fixed Log Spread: {asset_a} - beta x {asset_b}")
    axes[0].set_ylabel("Log spread")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(features.index, features["signal_zscore"])
    axes[1].axhline(1.5, linestyle="--")
    axes[1].axhline(-1.5, linestyle="--")
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_title("Lagged Trading Signal Z-Score")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Z-score")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_equity_comparison(results_by_name: dict[str, pd.DataFrame], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, results in results_by_name.items():
        ax.plot(results.index, results["cumulative_net_pnl"], label=name)
    ax.set_title("Out-of-Sample Cumulative Net PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative log-spread PnL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, path: str | Path, title: str) -> None:
    data = importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(data["feature"], data["importance"])
    ax.set_title(title)
    ax.set_xlabel("Absolute coefficient or feature importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_roc_curves(predictions: pd.DataFrame, model_names: list[str], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    for name in model_names:
        column = f"{name}_probability"
        if column not in predictions or predictions["actual"].nunique() < 2:
            continue
        RocCurveDisplay.from_predictions(predictions["actual"], predictions[column], name=name, ax=ax)
    ax.set_title("Out-of-Sample ROC Curves")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_synthetic_regime_performance(scenarios: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    regimes = list(scenarios["regime"].drop_duplicates())
    data = [scenarios.loc[scenarios["regime"] == regime, "ml_minus_baseline_pnl"].to_numpy() for regime in regimes]
    ax.boxplot(data, tick_labels=regimes, showmeans=True)
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title("Synthetic Regime Robustness: ML-Filtered Minus Baseline PnL")
    ax.set_xlabel("Synthetic regime")
    ax.set_ylabel("PnL difference")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scenario_spreads(sample_paths: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for scenario_id, group in sample_paths.groupby("scenario_id"):
        ax.plot(group["date"], group["spread"], label=f"Scenario {scenario_id}: {group['regime'].iloc[0]}")
    ax.set_title("Example Calibrated Synthetic Spread Paths")
    ax.set_xlabel("Synthetic date")
    ax.set_ylabel("Log spread")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
