# M6 Current Results and Significance

## Scope of the current run

The current reproducible M6 run uses deterministic synthetic daily prices for ten labeled assets. The synthetic generator is deliberately constructed with shared I(1) stochastic trends and stationary residual spreads for designated pairs. These ticker labels must not be interpreted as actual historical XOM, CVX, KO, PEP, JPM, or BAC prices.

## Pair-selection result

The training-only screen evaluated 45 possible pairs. Five passed the configured correlation, I(1), Engle-Granger, residual-ADF, and false-discovery controls. The selected synthetic-labeled pair was XOM-CVX. Its estimated training-period half-life was approximately 13.7 trading days.

**Inference:** the corrected synthetic generator and conservative screening controls now produce internally consistent cointegration diagnostics. This verifies the methodology but does not establish that the actual XOM-CVX pair is cointegrated over the same historical period.

## ML classification result

Random forest was selected before test evaluation because it achieved the highest validation F1. On the untouched synthetic test set its AUC was approximately 0.432 and its bootstrap confidence interval included 0.5. Logistic regression had a higher test AUC (approximately 0.540), but it was not the validation-selected winner.

**Inference:** the current synthetic evidence does not show stable out-of-sample classification superiority. Choosing logistic regression after observing test results would introduce model-selection bias.

## Trading result

The synthetic test-window baseline achieved cumulative net log-spread PnL of approximately 0.109 and Sharpe of 0.824. Logistic regression produced the highest ex-post Sharpe (approximately 0.988) and a smaller maximum drawdown, while the validation-selected random forest was close to the baseline. Confidence intervals are wide.

**Inference:** ML filtering can alter turnover and risk, but the present result is insufficient to claim a reliable profitability improvement. Classification metrics and economic metrics must be interpreted separately.

## Synthetic-regime result

The validation-selected random forest did not consistently outperform the baseline across calm, high-volatility, jump, stress, and weak-mean-reversion scenarios. It was relatively more helpful in weak-mean-reversion scenarios but generally failed to add value in calm, high-volatility, and stress conditions.

**Inference:** the robustness analysis rejects a universal claim that the current ML filter improves pairs trading. This negative finding is useful because it identifies the conditions under which further modeling and sensitivity analysis are necessary.

## What remains before final conclusions

The final empirical stage should run the strict public-data configuration, repeat pair screening across formation windows, perform cost/threshold/horizon/stop-loss sensitivity analysis, evaluate pair stability, and update the abstract and conclusion only after those genuine-data results are available.
