# Revised Preliminary Findings

These outputs are research-development results. They do not constitute investment advice or final evidence of profitability.

## Data and leakage controls
- Data source: `offline_reproducible_cointegrated_synthetic_daily_prices`.
- Synthetic data used: `True`.
- Pair selection was performed only through 2021-05-28.
- Purge horizon: 10 days; embargo: 2 days.

## Pair selection
- Pairs tested: 45; pairs passing all conservative filters: 4.
- Selected pair: CVX-XOM (passed_all_conservative_filters).
- Engle-Granger p-value: 0.0001; residual ADF p-value: 0.0000; FDR-adjusted p-value: 0.0023.

## ML validation and testing
- Model selection used validation data only: highest f1 on validation only.
- Selected model: gradient_boosting with validation F1 0.698 and test F1 0.454.
- Test positive-class prevalence: 0.314.

## Out-of-sample trading
- Best reported test-period strategy by Sharpe: ML-filtered (random_forest) with Sharpe -0.003.
- Classifier quality and trading profitability are reported separately; predictive metrics do not imply profitable execution.

## Synthetic robustness
- calm: ML outperformed baseline in 33.3% of scenarios; mean PnL difference -0.0023.
- high_volatility: ML outperformed baseline in 0.0% of scenarios; mean PnL difference -0.0012.
- jump: ML outperformed baseline in 0.0% of scenarios; mean PnL difference -0.0134.
- stress: ML outperformed baseline in 0.0% of scenarios; mean PnL difference 0.0000.
- weak_mean_reversion: ML outperformed baseline in 83.3% of scenarios; mean PnL difference 0.0125.

## Interpretation
The revised pipeline explicitly distinguishes synthetic from public data, freezes pair-selection and hedge-ratio estimates before the test period, purges overlapping label horizons, selects models on validation data, and reports test-period trading results with uncertainty intervals.
