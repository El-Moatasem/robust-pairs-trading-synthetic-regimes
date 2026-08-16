# Final M7 + ML/DL Results and Significance

## 1. Deterministic synthetic experiment

- Data are deterministic synthetic prices, not historical investment evidence.
- CVX-XOM passes the configured statistical screen.
- In the original ML benchmark, gradient boosting is selected by validation F1 and remains negative on the untouched synthetic test set.
- In the ML/DL extension, the GRU has the strongest validation F1 but reproduces the negative baseline PnL (about -0.0210) and Sharpe (about -0.154).
- No tested ML/DL model is both statistically passed and profitable on this synthetic fixed split.
- Synthetic regime tests do not show a universal ML advantage.

## 2. Full 10-asset genuine public fixed split

- Data source: included genuine adjusted-close public cache, 2018-01-02 through 2026-06-29; strict verifier reports `Synthetic data: False`.
- CVX-XOM is provisionally selected and fails the full fixed-split Engle-Granger/FDR screen.
- Random forest is validation-selected and produces test PnL about 0.0869 / Sharpe about 0.348 versus baseline PnL about -0.0230 / Sharpe about -0.084.
- Deep MLP also has positive PnL about 0.0704; GRU PnL is about 0.0058.
- These positive model paths are not labeled statistically validated because the pair is provisional.

## 3. Five-pair genuine public walk-forward: classical ML benchmark

The hypothesis family is fixed before test-period outcomes: CVX-XOM, KO-PEP, BAC-JPM, AAPL-MSFT, and QQQ-SPY (canonical orientation may reorder ticker names).

### Fold 1
- CVX-XOM passes all configured screens.
- EG p = 0.00427; residual ADF p = 0.0000395; BH-FDR p = 0.02133; half-life = 12.44 days.
- Baseline PnL = -0.18021.
- Gradient boosting is selected by validation F1 in the final three-model ML benchmark and reduces OOS PnL loss to -0.000835 / Sharpe -0.01044.

### Fold 2
- CVX-XOM again passes all configured 10% screens.
- EG p = 0.01063; residual ADF p = 0.000118; BH-FDR p = 0.05315; half-life = 16.97 days.
- Test period: 2023-01-06 to 2024-01-04.
- Random forest is selected by validation F1 in the ML-only benchmark.
- Baseline and selected-ML PnL = 0.100724; Sharpe = 0.901391; 12 trades; 58.33% trade win rate.
- This is the validation-selected ML benchmark's only pair/fold that both passes all configured screens and is profitable.

### Fold 3
- KO-PEP is provisional; no candidate passes every configured screen.
- Logistic regression is selected in the ML-only benchmark; PnL = -0.061162, same as the baseline.

## 4. Deep-learning extension on the five-pair walk-forward study

The final extension evaluates logistic regression, random forest, gradient boosting, a fixed three-hidden-layer deep MLP, and a 20-day sequence-aware PyTorch GRU under the same pair screen, costs, and leakage controls.

### All statistically passed + profitable model/pair cases

- Fold 1, CVX-XOM, random forest: PnL 0.047781 / Sharpe 0.734049.
- Fold 1, CVX-XOM, deep MLP: PnL 0.010307 / Sharpe 0.136817.
- Fold 2, CVX-XOM, logistic regression: PnL 0.100724 / Sharpe 0.901391.
- Fold 2, CVX-XOM, random forest: PnL 0.100724 / Sharpe 0.901391.
- Fold 2, CVX-XOM, gradient boosting: PnL 0.102858 / Sharpe 0.940556.
- Fold 2, CVX-XOM, deep MLP: PnL 0.100724 / Sharpe 0.901391.
- Fold 2, CVX-XOM, GRU: PnL 0.100724 / Sharpe 0.901391.

The unique real pair represented in all passed-and-profitable cases is therefore **CVX-XOM**, in folds 1 and/or 2 depending on the model.

### Validation-selected ML/DL system

Model choice uses validation F1 only:

1. Fold 1: gradient boosting, PnL -0.000835 / Sharpe -0.010442.
2. Fold 2: GRU, PnL 0.100724 / Sharpe 0.901391.
3. Fold 3: GRU, PnL -0.063813 / Sharpe -0.716321.

Under the pre-declared criterion (positive PnL and Sharpe in every fold, never below baseline, validation-only model selection), **no universal ML/DL solution is demonstrated**.

## 5. Strict statistical-gate fixed-model policy

A second policy stays flat whenever no pair passes every configured screen. This prevents trading the provisional fold-3 pair.

- Fixed random forest: aggregate observed gated PnL 0.148505; positive in both valid-pair folds; no trade in fold 3; zero negative observed folds.
- Fixed deep MLP: aggregate observed gated PnL 0.111031; positive in both valid-pair folds; no trade in fold 3; zero negative observed folds.

This is the strongest stability result in the current study. It is a promising candidate architecture, not proof of universal future profitability, because only two folds contain a statistically valid pair and external ETF replication remains pending.

## 6. Exhaustive 45-pair genuine public walk-forward screen

- 45 unordered pairs x 3 folds = 135 pair-fold hypotheses.
- BH-FDR is applied across all 45 hypotheses within each fold.
- Zero pair/fold cases pass every configured screen after this global correction.
- Some raw-diagnostic cases have positive OOS PnL (for example CVX-SPY fold 1 and CVX-XOM fold 2), but they are not called statistically validated because they fail the 45-pair FDR gate.

## 7. Independent backtest and sensitivity

- The independent trade-replay engine reproduces the primary fixed-split M7 trade counts and PnL to numerical precision.
- Trading-rule, cost, significance, and subperiod sensitivity show strong regime dependence and do not justify post-hoc parameter selection.

## 8. Economically pre-specified ETF replication

The final source includes a strict real-data replication for SPY-IVV, SPY-VOO, IVV-VOO, QQQ-QQQM, GLD-IAU, IWM-VTWO, VTI-ITOT, and AGG-BND, including ML/DL walk-forward code. No ETF numerical result is asserted in the packaged report because a verified ETF cache was not available in the artifact-build environment. Strict mode fails rather than substituting synthetic data.

## Final conclusion

The project demonstrates genuine historically profitable, statistically passed CVX-XOM cases, and the deep-learning extension adds further profitable model-filtered cases. It does **not** demonstrate a universal validation-selected ML/DL alpha engine. The strongest current research design is a strict statistical gate plus a fixed nonlinear filter: trade statistically valid opportunities and remain flat when the screen fails, while requiring additional external and longer-horizon validation before production deployment.

## Advanced ML/DL result update

The final advanced extension is documented in `docs/ADVANCED_LEARNING_RESULTS.md`. The key result is that additional model complexity does not produce a universal solution. The fold-2 regime-aware mixture-of-experts achieves PnL `0.163822` and Sharpe `1.589911`, but loses in fold 1. The strict statistical-gate fixed random forest and deep MLP remain the only tested fixed policies that are positive in both statistically valid CVX-XOM folds and flat in the rejected third fold.
