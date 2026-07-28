from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from src.data import generate_synthetic_prices
from src.features import MODEL_FEATURE_COLUMNS, build_feature_frame
from src.labels import build_convergence_labels
from src.pairs import screen_pairs, select_top_pair
from src.splits import make_time_split
from src.utils import load_config


def _config() -> dict:
    return load_config("config/config.yaml")


def test_synthetic_generator_has_conservative_cointegrated_pair() -> None:
    cfg = _config()
    prices = generate_synthetic_prices(cfg["data"]["tickers"], n_days=1800, seed=42, start="2018-01-01")
    split = make_time_split(prices.index, cfg)
    candidates, summary = screen_pairs(prices.loc[split.train_index], cfg)
    assert summary["pairs_tested"] > 0
    assert summary["pairs_passing_all"] >= 1
    assert bool(candidates.iloc[0]["passes_all"])
    assert candidates.iloc[0]["eg_pvalue"] < cfg["pair_selection"]["max_eg_pvalue"]
    assert candidates.iloc[0]["residual_adf_pvalue"] < cfg["pair_selection"]["max_residual_adf_pvalue"]


def test_future_price_changes_do_not_change_past_predictors() -> None:
    cfg = _config()
    prices = generate_synthetic_prices(cfg["data"]["tickers"], n_days=1000, seed=7, start="2018-01-01")
    split = make_time_split(prices.index, cfg)
    candidates, _ = screen_pairs(prices.loc[split.train_index], cfg)
    a, b, alpha, beta, _ = select_top_pair(candidates)
    base = build_feature_frame(prices, a, b, cfg, alpha, beta)
    cutoff = prices.index[700]
    changed = prices.copy()
    changed.loc[changed.index > cutoff, a] *= 4.0
    changed_features = build_feature_frame(changed, a, b, cfg, alpha, beta)
    common = base.index.intersection(changed_features.index)
    past = common[common <= cutoff]
    pd.testing.assert_frame_equal(base.loc[past, MODEL_FEATURE_COLUMNS], changed_features.loc[past, MODEL_FEATURE_COLUMNS])


def test_labels_are_created_only_at_entry_signals() -> None:
    cfg = _config()
    prices = generate_synthetic_prices(cfg["data"]["tickers"], n_days=1000, seed=9, start="2018-01-01")
    split = make_time_split(prices.index, cfg)
    candidates, _ = screen_pairs(prices.loc[split.train_index], cfg)
    a, b, alpha, beta, _ = select_top_pair(candidates)
    features = build_feature_frame(prices, a, b, cfg, alpha, beta)
    labels = build_convergence_labels(features, cfg)
    entry = float(cfg["labels"]["entry_zscore"])
    assert len(labels) > 0
    assert (features.loc[labels.index, "signal_zscore"].abs() >= entry).all()


def test_purge_and_embargo_create_disjoint_time_sets() -> None:
    cfg = _config()
    index = pd.bdate_range("2020-01-01", periods=1000)
    split = make_time_split(index, cfg)
    assert set(split.train_index).isdisjoint(set(split.validation_index))
    assert set(split.validation_index).isdisjoint(set(split.test_index))
    assert max(split.train_index) < min(split.validation_index) < max(split.validation_index) < min(split.test_index)
