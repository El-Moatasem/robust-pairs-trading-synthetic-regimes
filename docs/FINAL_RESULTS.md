# Final M7 Results and Significance

## Fixed synthetic experiment

- Deterministic synthetic data; no historical-price claim.
- CVX-XOM passes the conservative pair screen.
- Gradient boosting is selected by validation F1 in the final deterministic run.
- Test-period ML variants do not establish a robust advantage over the baseline, and regime tests show that ML does not consistently outperform across calm, high-volatility, jump, weak-mean-reversion and stress scenarios.

## Full 10-asset public fixed split

- Data source: included Yahoo Finance adjusted-close cache, 2018-01-02 through 2026-06-29.
- CVX-XOM is provisionally selected.
- It does not pass the full-sample Engle-Granger/FDR discovery screen.
- Validation-selected random forest: test AUC about 0.553; cumulative net PnL about 0.0869 vs -0.0230 baseline; Sharpe about 0.348 vs -0.084 baseline.
- PnL and Sharpe confidence intervals include zero, so the fixed-split improvement is economically interesting but statistically inconclusive.

## M7 five-pair walk-forward analysis

### Fold 1

- Selected pair: CVX-XOM, passed all configured discovery screens.
- EG p = 0.00427, residual ADF p = 0.0000395, BH-FDR p = 0.02133, half-life = 12.44 days.
- Baseline PnL = -0.1802; selected-ML PnL = -0.0803.
- ML materially reduces the loss, but the fold is not profitable.

### Fold 2

- Selected pair: CVX-XOM, passed all configured discovery screens at the exploratory 10% level.
- EG p = 0.01063, residual ADF p = 0.000118, BH-FDR p = 0.05315, half-life = 16.97 days.
- Test period: 2023-01-06 to 2024-01-04.
- Baseline and selected-ML net PnL = 0.10072; Sharpe = 0.90139; 12 trades; 58.33% trade win rate.
- This is the only walk-forward fold in the reported discovery universe that both passes all configured screens and is profitable. The ML filter does not add incremental PnL in this fold because it accepts the same trades as the baseline.

### Fold 3

- Selected pair: KO-PEP, provisional; no pair passes every configured discovery screen.
- Baseline and selected-ML PnL = -0.0612.

## Independent backtest

The independent trade-replay engine exactly reproduces the primary fixed-split M7 backtest within floating-point precision:

- baseline: 35 trades in both engines; absolute PnL difference approximately 5.6e-17;
- selected logistic model: 32 trades in both engines; absolute PnL difference approximately 2.8e-17.

## Trading-rule and subperiod sensitivity

The M7 constrained fixed-split sensitivity does not reveal a robust positive strategy. Some settings reduce the loss relative to baseline, but the selected-ML PnL remains negative. Within the fixed public test period, only the final chronological subperiod is positive, while the first two are negative. This supports a regime-dependent rather than universal-profit interpretation.

## Overall conclusion

The final evidence supports the software/research framework more strongly than it supports a universal trading edge. CVX-XOM provides one statistically screened and profitable walk-forward period, but profitability is not stable across folds and the profit in that successful fold is not created by incremental ML filtering. ML appears more credible as a signal-quality and risk-control layer than as a guaranteed alpha generator.


## Exhaustive 45-pair real-data walk-forward screen

The original 10-ticker real-data universe contains 45 unordered pairs. The final package additionally screens all 45 pairs in each of three walk-forward folds, for 135 pair-fold hypotheses. BH-FDR is applied across the full 45-pair hypothesis family inside each fold. **No pair passes every configured screen after this global correction.**

The exploratory raw-diagnostic backtest table contains 10 cases that pass correlation, Engle-Granger, residual ADF, and I(1) diagnostics before the global FDR gate. Six have positive selected-ML OOS PnL; examples include CVX-SPY (Fold 1, PnL 0.2974, Sharpe 1.0188) and CVX-XOM (Fold 2, PnL 0.1007, Sharpe 0.9014). These are not called statistically validated profitable pairs because their 45-pair BH-FDR p-values exceed 0.10.

This comparison is central to the final interpretation: the five-pair economically restricted study and the 45-pair broad discovery study answer different hypothesis-testing questions, and the report preserves that distinction instead of choosing the scope after observing test profitability.

## Economically pre-specified ETF replication

The final source includes a strict public-data replication across eight economically motivated ETF pair hypotheses: SPY-IVV, SPY-VOO, IVV-VOO, QQQ-QQQM, GLD-IAU, IWM-VTWO, VTI-ITOT, and AGG-BND. The experiment reuses the same leakage controls, pair diagnostics, FDR, model-selection procedure, costs, walk-forward analysis, independent backtest, EDA, and sensitivity framework.

No ETF numerical result is asserted in this packaged report because `data/raw/etf_prices.csv` was not available in the artifact-build environment and outbound market-data access was unavailable. Strict mode fails rather than substituting synthetic data. Run the documented ETF commands in a network-enabled environment and include numerical claims only after verification reports `Synthetic data: False`.
