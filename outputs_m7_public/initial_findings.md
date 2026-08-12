# Revised Preliminary Findings

These outputs are research-development results. They do not constitute investment advice or final evidence of profitability.

## Data and leakage controls
- Data source: `cached_public_adjusted_close:data/raw/public_prices.csv`.
- Synthetic data used: `False`.
- Pair selection was performed only through 2022-03-11.
- Purge horizon: 10 days; embargo: 2 days.

## Pair selection
- Pairs tested: 5; pairs passing all conservative filters: 0.
- Selected pair: KO-PEP (provisional_best_available_failed_one_or_more_filters).
- Engle-Granger p-value: 0.1092; residual ADF p-value: 0.0026; FDR-adjusted p-value: 0.5459.

## ML validation and testing
- Model selection used validation data only: highest f1 on validation only.
- Selected model: logistic_regression with validation F1 0.447 and test F1 0.395.
- Test positive-class prevalence: 0.283.

## Out-of-sample trading
- Best reported test-period strategy by Sharpe: ML-filtered (gradient_boosting) with Sharpe -0.279.
- Classifier quality and trading profitability are reported separately; predictive metrics do not imply profitable execution.

## Synthetic robustness
- calm: ML outperformed baseline in 50.0% of scenarios; mean PnL difference 0.0007.
- high_volatility: ML outperformed baseline in 16.7% of scenarios; mean PnL difference -0.0034.
- jump: ML outperformed baseline in 50.0% of scenarios; mean PnL difference -0.0101.
- stress: ML outperformed baseline in 83.3% of scenarios; mean PnL difference 0.0644.
- weak_mean_reversion: ML outperformed baseline in 50.0% of scenarios; mean PnL difference 0.0104.

## Interpretation
The revised pipeline explicitly distinguishes synthetic from public data, freezes pair-selection and hedge-ratio estimates before the test period, purges overlapping label horizons, selects models on validation data, and reports test-period trading results with uncertainty intervals.
