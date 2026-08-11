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


def test_candidate_pair_restriction_is_honored() -> None:
    cfg = _config()
    cfg = copy.deepcopy(cfg)
    cfg["pair_selection"]["candidate_pairs"] = ["XOM-CVX", "KO-PEP"]
    prices = generate_synthetic_prices(cfg["data"]["tickers"], n_days=1000, seed=42, start="2018-01-01")
    split = make_time_split(prices.index, cfg)
    candidates, summary = screen_pairs(prices.loc[split.train_index], cfg)
    assert summary["pairs_tested"] == 2
    observed = {f"{row.asset_a}-{row.asset_b}" for row in candidates.itertuples()}
    assert observed.issubset({"CVX-XOM", "KO-PEP"})


def test_independent_trade_replay_matches_primary_backtest() -> None:
    from src.backtest import run_backtest
    from src.independent_backtest import compare_backtest_engines, replay_trade_backtest

    cfg = _config()
    prices = generate_synthetic_prices(cfg["data"]["tickers"], n_days=1000, seed=13, start="2018-01-01")
    split = make_time_split(prices.index, cfg)
    candidates, _ = screen_pairs(prices.loc[split.train_index], cfg)
    a, b, alpha, beta, _ = select_top_pair(candidates)
    features = build_feature_frame(prices, a, b, cfg, alpha, beta)
    test_features = features.loc[features.index.intersection(split.test_index)]
    _, primary_trades = run_backtest(test_features, cfg, "primary")
    independent_trades, _ = replay_trade_backtest(test_features, cfg)
    comparison = compare_backtest_engines(primary_trades, independent_trades)
    assert comparison["trade_count_match"]
    assert comparison["pnl_match_within_1e_10"]


def test_pair_screening_is_invariant_to_input_column_order() -> None:
    """Cached CSV column order must not change candidate-pair regression orientation."""
    cfg = _config()
    cfg = copy.deepcopy(cfg)
    cfg["pair_selection"]["candidate_pairs"] = ["XOM-CVX", "KO-PEP"]
    prices = generate_synthetic_prices(cfg["data"]["tickers"], n_days=1000, seed=42, start="2018-01-01")
    split = make_time_split(prices.index, cfg)
    train = prices.loc[split.train_index]
    first, _ = screen_pairs(train, cfg)
    reversed_cols = train[list(reversed(train.columns))]
    second, _ = screen_pairs(reversed_cols, cfg)
    cols = ["asset_a", "asset_b", "eg_pvalue", "residual_adf_pvalue", "estimated_half_life_days"]
    a = first[cols].sort_values(["asset_a", "asset_b"]).reset_index(drop=True)
    b = second[cols].sort_values(["asset_a", "asset_b"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, rtol=1e-12, atol=1e-12)


def test_screen_pairs_return_all_exposes_full_unordered_universe():
    from src.pairs import screen_pairs
    cfg = _config()
    cfg['data']['mode'] = 'synthetic'
    cfg['data']['tickers'] = ['A', 'B', 'C', 'D']
    cfg['pair_selection']['top_n'] = 2
    cfg['pair_selection'].pop('candidate_pairs', None)
    idx = pd.date_range('2020-01-01', periods=300, freq='B')
    rng = np.random.default_rng(123)
    base = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
    prices = pd.DataFrame({
        'A': base,
        'B': base * np.exp(rng.normal(0, 0.01, len(idx))),
        'C': 80 * np.exp(np.cumsum(rng.normal(0, 0.012, len(idx)))),
        'D': 120 * np.exp(np.cumsum(rng.normal(0, 0.009, len(idx)))),
    }, index=idx)
    table, summary = screen_pairs(prices, cfg, return_all=True)
    assert len(table) == 6
    assert summary['pairs_tested'] == 6


def test_etf_replication_is_strict_public_and_prespecified():
    cfg = load_config('config/config_m7_etf.yaml')
    assert cfg['data']['mode'] == 'public'
    pairs = cfg['pair_selection']['candidate_pairs']
    assert len(pairs) == 8
    assert cfg['pair_selection']['top_n'] >= len(pairs)
    assert 'SPY-IVV' in pairs and 'QQQ-QQQM' in pairs and 'GLD-IAU' in pairs
