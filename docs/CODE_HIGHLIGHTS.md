# Important Code Sections for the Report

| Module | Research control implemented |
|---|---|
| `src/data.py` | Explicit synthetic/public modes and a correct shared-trend plus stationary-residual generator. |
| `src/splits.py` | Purged and embargoed chronological train/validation/test partitions. |
| `src/pairs.py` | Training-only I(1), correlation, Engle-Granger, residual ADF, FDR, OLS intercept, hedge ratio, and half-life. |
| `src/features.py` | One-day-lagged predictors, exact persistence measure, trailing stress proxy, and fixed training-period spread definition. |
| `src/labels.py` | Signal-only labels based on convergence-before-stop and four-leg pair-trading costs. |
| `src/models.py` | Validation-only threshold and model selection; untouched test metrics and AUC intervals. |
| `src/backtest.py` | Test-only baseline and filtered strategies, two-leg turnover costs, trade ledger, and block-bootstrap intervals. |
| `src/synthetic.py` | Training-calibrated OU/AR(1) regimes and scenario-by-scenario economic evaluation. |
| `tests/test_research_controls.py` | Tests for conservative cointegration, feature leakage, signal labels, and split separation. |
