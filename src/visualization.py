from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import RocCurveDisplay


def plot_pair_diagnostics(features: pd.DataFrame, path: str | Path, asset_a: str, asset_b: str) -> None:
    """Generates and saves a two-panel diagnostic plot for a selected asset pair.

    Plots the log-spread time series in the top subplot and the lagged trading signal
    z-score with entry/exit threshold lines in the bottom subplot.

    Args:
        features (pd.DataFrame): DataFrame containing `spread` and `signal_zscore` time series.
        path (str | Path): Destination file path where the plot figure will be saved.
        asset_a (str): Ticker symbol for the primary asset in the pair.
        asset_b (str): Ticker symbol for the secondary asset in the pair.
    """
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
    """Plots out-of-sample cumulative net PnL performance curves across strategies.

    Overlaying multiple strategy equity curves over time to compare backtest results.

    Args:
        results_by_name (dict[str, pd.DataFrame]): A dictionary mapping strategy names to
            their backtest result DataFrames containing a `cumulative_net_pnl` column.
        path (str | Path): Destination file path where the plot figure will be saved.
    """
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
    """Generates and saves a horizontal bar plot of feature importances or model coefficients.

    Args:
        importance (pd.DataFrame): DataFrame containing `feature` names and their corresponding
            `importance` metric values.
        path (str | Path): Destination file path where the plot figure will be saved.
        title (str): Title header text for the generated plot.
    """
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
    """Plots out-of-sample Receiver Operating Characteristic (ROC) curves for candidate models.

    Args:
        predictions (pd.DataFrame): DataFrame containing true binary outcomes in the `actual`
            column and predicted probabilities in `{model_name}_probability` columns.
        model_names (list[str]): List of model identifier names to include in the plot.
        path (str | Path): Destination file path where the plot figure will be saved.
    """
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
    """Creates a boxplot comparing net PnL differences across synthetic stress regimes.

    Visualizes the distribution of performance improvements (ML-filtered strategy minus baseline PnL)
    grouped by market regime.

    Args:
        scenarios (pd.DataFrame): Scenario backtest results DataFrame containing `regime` and
            `ml_minus_baseline_pnl` columns.
        path (str | Path): Destination file path where the plot figure will be saved.
    """
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
    """Plots representative calibrated synthetic log-spread sample paths across scenarios.

    Args:
        sample_paths (pd.DataFrame): DataFrame containing simulated scenario paths with columns
            `scenario_id`, `date`, `spread`, and `regime`.
        path (str | Path): Destination file path where the plot figure will be saved.
    """
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
