from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_walk_forward_pnl(summary: pd.DataFrame, output_dir: Path) -> Path:
    """Plot baseline vs validation-selected ML PnL for each walk-forward fold."""
    frame = summary.copy()
    x = np.arange(len(frame))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, frame["baseline_pnl"], width, label="Classical baseline")
    ax.bar(x + width / 2, frame["ml_pnl"], width, label="Validation-selected ML")
    ax.axhline(0.0, linewidth=1.0)
    labels = [f"Fold {int(r.fold)}\n{r.selected_pair}" for r in frame.itertuples()]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Out-of-sample net log-spread PnL")
    ax.set_xlabel("Walk-forward fold and selected pair")
    ax.set_title("Walk-Forward Out-of-Sample Trading Performance")
    ax.legend()
    return _save(fig, output_dir / "walk_forward_pnl.png")


def plot_walk_forward_diagnostics(summary: pd.DataFrame, output_dir: Path) -> Path:
    """Plot fold-level EG/FDR p-values and configured 10% significance threshold."""
    frame = summary.copy()
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(x, frame["eg_pvalue"], marker="o", label="Engle-Granger p-value")
    ax.plot(x, frame["eg_fdr_pvalue"], marker="o", label="BH-FDR adjusted p-value")
    ax.axhline(0.10, linestyle="--", linewidth=1.0, label="10% screening threshold")
    labels = [f"Fold {int(r.fold)}\n{r.selected_pair}" for r in frame.itertuples()]
    ax.set_xticks(x, labels)
    ax.set_ylim(bottom=0.0, top=max(0.35, float(frame[["eg_pvalue", "eg_fdr_pvalue"]].max().max()) * 1.15))
    ax.set_ylabel("p-value")
    ax.set_xlabel("Walk-forward fold and selected pair")
    ax.set_title("Walk-Forward Cointegration Screening Stability")
    ax.legend()
    return _save(fig, output_dir / "walk_forward_cointegration_diagnostics.png")


def plot_trading_rule_sensitivity(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot ML-minus-baseline PnL across trading-rule sensitivity scenarios."""
    data = frame.copy()
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.bar(data["scenario"], data["ml_minus_baseline_pnl"])
    ax.axhline(0.0, linewidth=1.0)
    ax.set_ylabel("ML minus baseline net PnL")
    ax.set_xlabel("Sensitivity scenario")
    ax.set_title("Trading-Rule and Cost Sensitivity")
    ax.tick_params(axis="x", rotation=35)
    return _save(fig, output_dir / "trading_rule_sensitivity.png")


def plot_subperiod_stability(frame: pd.DataFrame, output_dir: Path) -> Path:
    """Plot baseline and selected-ML PnL across chronological public-test subperiods."""
    data = frame.copy()
    x = np.arange(len(data))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - width / 2, data["baseline_pnl"], width, label="Classical baseline")
    ax.bar(x + width / 2, data["ml_pnl"], width, label="Validation-selected ML")
    ax.axhline(0.0, linewidth=1.0)
    labels = [f"Subperiod {int(r.subperiod)}\n{r.start[:4]}-{r.end[:4]}" for r in data.itertuples()]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Net log-spread PnL")
    ax.set_xlabel("Chronological test subperiod")
    ax.set_title("Public-Test Subperiod Stability")
    ax.legend()
    return _save(fig, output_dir / "subperiod_stability.png")
