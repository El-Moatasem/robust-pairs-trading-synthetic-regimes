# Peer Review Response Matrix

| Peer-review issue | Revision implemented |
|---|---|
| Daily data and convergence horizon need theoretical linkage | Added half-life estimation, explicit convergence horizon, and documented the limitation that cointegration does not guarantee convergence within a fixed horizon. |
| Cointegration may decay or break | Pair selection is training-only; the frozen relationship is evaluated out of sample; synthetic weak-mean-reversion and stress regimes are included. |
| Regime-switching theory was unclear | Clarified that the project uses calibrated synthetic regimes rather than a formal hidden Markov model. Added an AR(1)/OU calibration and explicit regime multipliers. |
| Regime-stress proxy was undefined | Added an explicit formula based on trailing percentiles of spread volatility and weakening correlation. |
| Deviation persistence was undefined | Defined it as the lagged consecutive count of observations above a configured absolute z-score threshold. |
| Stop-loss rule was unclear | Added a precise absolute z-score stop, convergence exit, and maximum holding rule in both labels and backtests. |
| Number of pairs and rejection criteria were missing | `candidate_pairs.csv` now reports every displayed diagnostic and `pair_selection_summary.json` reports pairs tested and pairs passing all filters. |
| Features lacked mathematical definitions | Added `docs/METHODOLOGY.md` and `feature_definitions.csv`. |
| Backtest was high level | Added explicit entry, exit, stop, holding, signal timing, PnL timing, pair-leg costs, trade ledger, and uncertainty metrics. |
| Sharpe risk-free rate was missing | Added `annual_risk_free_rate` to the configuration and output table. |
| Synthetic calibration was not explained | Added training-spread AR(1)/OU calibration and saved parameters to `synthetic_calibration.json`. |
| Baseline performance was not interpreted | Revised findings compare baseline and every ML-filtered strategy and state that predictive metrics do not imply profitability. |
| In-sample versus out-of-sample metrics were unclear | All metric columns are explicitly prefixed with `validation_` or `test_`; model selection uses validation only. |
| Synthetic and historical data could be confused | Data mode and synthetic status are saved in `data_summary.json` and reported in `initial_findings.md`; strict public mode fails instead of silently falling back. |
| Engle-Granger and residual ADF results could disagree | The generator was corrected to use stationary residuals; both diagnostics are reported and can be required simultaneously. |
| Pair screening could leak future information | Pair selection, FDR correction, intercept, and hedge ratio are all estimated only in the training window. |
| Label horizons could overlap split boundaries | Added purge and embargo gaps. |
| Multiple-pair screening creates data-snooping risk | Added Benjamini-Hochberg FDR correction. |
| Model selection and trading model were inconsistent | Every model is backtested on the same test period; the pre-selected validation winner is clearly identified separately from ex-post test rankings. |
| Statistical uncertainty was absent | Added bootstrap AUC intervals and block-bootstrap PnL and Sharpe intervals. |
