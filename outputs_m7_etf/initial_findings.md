# Revised Preliminary Findings

These outputs are research-development results. They do not constitute investment advice or final evidence of profitability.

## Data and leakage controls
- Data source: `cached_public_adjusted_close:data/raw/etf_prices.csv`.
- Synthetic data used: `False`.
- Pair selection was performed only through 2023-08-03.
- Purge horizon: 10 days; embargo: 2 days.

## Pair selection
- Pairs tested: 8; pairs passing all conservative filters: 2.
- Selected pair: IVV-VOO (passed_all_conservative_filters).
- Engle-Granger p-value: 0.0000; residual ADF p-value: 0.0000; FDR-adjusted p-value: 0.0000.

## ML validation and testing
- Model selection used validation data only: highest f1 on validation only.
- Selected model: dummy_majority with validation F1 0.000 and test F1 0.000.
- Test positive-class prevalence: 0.000.

## Out-of-sample trading
- Best reported test-period strategy by Sharpe: ML-filtered (dummy_majority) with Sharpe 0.000.
- Classifier quality and trading profitability are reported separately; predictive metrics do not imply profitable execution.

## Synthetic robustness
- calm: ML outperformed baseline in 100.0% of scenarios; mean PnL difference 0.0107.
- high_volatility: ML outperformed baseline in 100.0% of scenarios; mean PnL difference 0.0108.
- jump: ML outperformed baseline in 100.0% of scenarios; mean PnL difference 0.0067.
- stress: ML outperformed baseline in 83.3% of scenarios; mean PnL difference 0.0017.
- weak_mean_reversion: ML outperformed baseline in 100.0% of scenarios; mean PnL difference 0.0080.

## Interpretation
The revised pipeline explicitly distinguishes synthetic from public data, freezes pair-selection and hedge-ratio estimates before the test period, purges overlapping label horizons, selects models on validation data, and reports test-period trading results with uncertainty intervals.
