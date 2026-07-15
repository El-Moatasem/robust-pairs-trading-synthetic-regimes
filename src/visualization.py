from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(results: pd.DataFrame, path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    results["cum_pnl"].plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_scenario_volatility(summary: pd.DataFrame, path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(summary["std"], bins=12)
    ax.set_title("Synthetic Scenario Volatility Distribution")
    ax.set_xlabel("Spread standard deviation")
    ax.set_ylabel("Scenario count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importance: pd.DataFrame, path: str | Path, title: str) -> None:
    data = importance.sort_values("importance", ascending=True).tail(9)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(data["feature"], data["importance"])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
