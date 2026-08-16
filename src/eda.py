from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .features import MODEL_FEATURE_COLUMNS


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def generate_eda_figures(prices: pd.DataFrame, features: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Create appendix-ready EDA figures with explicit axes, labels, and scales."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    normalized = prices / prices.iloc[0] * 100.0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for column in normalized.columns:
        ax.plot(normalized.index, normalized[column], label=column, linewidth=1.0)
    ax.set_title("Normalized Public-Market Prices (Start = 100)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Normalized price index")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(True, alpha=0.25)
    path = out / "eda_normalized_prices.png"; _save(fig, path); paths.append(path)

    returns = np.log(prices).diff().dropna()
    corr = returns.corr()
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(corr.to_numpy(), vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_title("Daily Log-Return Correlation Matrix")
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)), corr.index)
    fig.colorbar(image, ax=ax, label="Correlation")
    path = out / "eda_return_correlation_heatmap.png"; _save(fig, path); paths.append(path)

    selected = features["asset_a"].iloc[0], features["asset_b"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(features["spread"].dropna(), bins=40, edgecolor="black", linewidth=0.5)
    ax.set_title(f"Spread Distribution: {selected[0]}-{selected[1]}")
    ax.set_xlabel("Log spread")
    ax.set_ylabel("Frequency")
    ax.grid(True, axis="y", alpha=0.25)
    path = out / "eda_spread_distribution.png"; _save(fig, path); paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(features.index, features["rolling_corr"])
    ax.axhline(0.0, linewidth=0.8)
    ax.set_title(f"Lagged Rolling Return Correlation: {selected[0]}-{selected[1]}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling correlation")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=0.25)
    path = out / "eda_rolling_correlation.png"; _save(fig, path); paths.append(path)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(features.index, features["half_life"])
    ax.set_title(f"Lagged Rolling Mean-Reversion Half-Life: {selected[0]}-{selected[1]}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Estimated half-life (trading days)")
    ax.grid(True, alpha=0.25)
    path = out / "eda_rolling_half_life.png"; _save(fig, path); paths.append(path)

    melted = features[MODEL_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    # Standardize only for visualization so features with different units can share an axis.
    standardized = (melted - melted.mean()) / melted.std(ddof=0).replace(0.0, np.nan)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.boxplot([standardized[col].dropna().clip(-5, 5) for col in MODEL_FEATURE_COLUMNS], tick_labels=MODEL_FEATURE_COLUMNS)
    ax.set_title("Standardized ML Feature Distributions (Clipped to +/-5 SD for Display)")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Standard deviations from feature mean")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    path = out / "eda_feature_distributions.png"; _save(fig, path); paths.append(path)

    return paths
