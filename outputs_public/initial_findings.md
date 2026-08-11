# Revised Preliminary Findings

These outputs are research-development results. They do not constitute investment advice or final evidence of profitability.

## Data and leakage controls
- Data source: `cached_public_adjusted_close:data/raw/public_prices.csv`.
- Synthetic data used: `False`.
- Pair selection was performed only through 2022-03-11.
- Purge horizon: 10 days; embargo: 2 days.

## Pair selection
- Pairs tested: 45; pairs passing all conservative filters: 0.
- Selected pair: CVX-XOM (provisional_best_available_failed_one_or_more_filters).
- Engle-Granger p-value: 0.6360; residual ADF p-value: 0.0703; FDR-adjusted p-value: 0.9827.

## ML validation and testing
- Model selection used validation data only: highest f1 on validation only.
- Selected model: random_forest with validation F1 0.444 and test F1 0.364.
- Test positive-class prevalence: 0.215.

## Out-of-sample trading
- Best reported test-period strategy by Sharpe: ML-filtered (random_forest) with Sharpe 0.348.
- Classifier quality and trading profitability are reported separately; predictive metrics do not imply profitable execution.

## Synthetic robustness
- calm: ML outperformed baseline in 33.3% of scenarios; mean PnL difference 0.0029.
- high_volatility: ML outperformed baseline in 0.0% of scenarios; mean PnL difference -0.0731.
- jump: ML outperformed baseline in 50.0% of scenarios; mean PnL difference -0.0503.
- stress: ML outperformed baseline in 66.7% of scenarios; mean PnL difference 0.0860.
- weak_mean_reversion: ML outperformed baseline in 33.3% of scenarios; mean PnL difference -0.0108.

## Interpretation
The revised pipeline explicitly distinguishes synthetic from public data, freezes pair-selection and hedge-ratio estimates before the test period, purges overlapping label horizons, selects models on validation data, and reports test-period trading results with uncertainty intervals.
