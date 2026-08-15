# Changelog

## Revised peer-review-response version

- Corrected the synthetic generator so designated pairs have stationary cointegrating residuals.
- Added strict public-data mode and explicit synthetic-data reporting.
- Added training-only pair selection and fixed hedge-ratio estimation.
- Added I(1) checks, residual ADF diagnostics, and FDR correction.
- Added one-day feature lags and removed full-sample rank leakage.
- Defined deviation persistence and regime-stress features.
- Added signal-only economic labels with correct four-leg round-trip costs.
- Added purged and embargoed three-way chronological splits.
- Added validation-only model and threshold selection.
- Backtested all models on the same untouched test period.
- Added confusion matrix, Brier score, class prevalence, and AUC intervals.
- Added trade ledgers, risk-free-rate configuration, four-leg execution costs, and block-bootstrap PnL/Sharpe intervals.
- Added calibrated synthetic regime evaluation.
- Added peer-review response, mathematical methodology, tests, output verification, and final report support.

## Final advanced ML/DL robustness extension
- Added a compact LSTM sequence classifier using the same leakage-safe 20-day lagged feature window as the GRU.
- Added a causal temporal CNN (TCN) with dilated one-dimensional convolutions.
- Added a training-defined regime mixture-of-experts policy with validation-only expert selection.
- Added an uncertainty-aware consensus/abstention policy with validation-only vote-strength selection.
- Added strict-universality and observed-gated-robustness tests that never use test PnL for model/policy selection.
- Added advanced synthetic, public fixed-split and genuine public walk-forward output tables and figures.
- Expanded automated controls to 13 passing tests.
