from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_equity_curve(backtest_df: pd.DataFrame, path: str | Path, title: str = "Equity Curve") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    backtest_df["equity_curve"].plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative PnL")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scenario_distribution(summary_df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    summary_df["std"].plot(kind="hist", bins=15, ax=ax)
    ax.set_title("Synthetic Scenario Volatility Distribution")
    ax.set_xlabel("Spread standard deviation")
    ax.set_ylabel("Scenario count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
