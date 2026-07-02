from src.data import DataConfig, get_price_data
from src.pairs import screen_candidate_pairs
from src.features import construct_pair_features
from src.backtest import generate_baseline_signals, backtest_pair_strategy
from src.synthetic import generate_synthetic_scenarios


def test_pipeline_smoke():
    prices = get_price_data(DataConfig(tickers=["AAA", "AAB", "BBB", "BBC"], use_sample_data=True))
    pairs = screen_candidate_pairs(prices, min_abs_correlation=0.1, max_cointegration_pvalue=1.0, top_n=2)
    assert not pairs.empty
    p = pairs.iloc[0]
    features = construct_pair_features(prices, p.asset_a, p.asset_b)
    signals = generate_baseline_signals(features)
    _, metrics = backtest_pair_strategy(signals)
    assert "cumulative_pnl" in metrics


def test_synthetic_scenarios():
    scenarios = generate_synthetic_scenarios(n_scenarios=5, n_steps=100)
    assert len(scenarios) == 5
