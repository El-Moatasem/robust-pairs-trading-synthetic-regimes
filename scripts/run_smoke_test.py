from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data import DataConfig, get_price_data
from src.pairs import screen_candidate_pairs
from src.features import construct_pair_features
from src.backtest import generate_baseline_signals, backtest_pair_strategy
from src.labels import build_convergence_labels, get_model_dataset
from src.models import train_trade_filters
from src.synthetic import generate_synthetic_scenarios

prices = get_price_data(DataConfig(tickers=["AAA", "AAB", "BBB", "BBC"], use_sample_data=True, random_state=42))
pairs = screen_candidate_pairs(prices, min_abs_correlation=0.1, max_cointegration_pvalue=1.0, top_n=2)
assert not pairs.empty
pair = pairs.iloc[0]
features = construct_pair_features(prices, pair.asset_a, pair.asset_b)
signals = generate_baseline_signals(features)
bt, metrics = backtest_pair_strategy(signals)
assert "sharpe" in metrics
labeled = build_convergence_labels(signals)
X, y = get_model_dataset(labeled)
if len(X) > 10 and y.nunique() > 1:
    results = train_trade_filters(X, y, model_names=["logistic_regression", "random_forest"])
    assert results
scenarios = generate_synthetic_scenarios(n_scenarios=3, n_steps=50)
assert len(scenarios) == 3
print("Smoke test passed.")
