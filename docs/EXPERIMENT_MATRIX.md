# Final Experiment Matrix

| Experiment | Data | Hypothesis scope | Main purpose | Reproducible output |
|---|---|---|---|---|
| Deterministic synthetic pipeline | Synthetic | 45 synthetic pair combinations | Software/methodology validation and controlled regimes | `outputs/` |
| Original fixed public discovery | Real Yahoo Finance cache | 45 unordered pairs from 10 tickers | Historical fixed-split benchmark | `outputs_public/` |
| M7 economic-pair walk-forward | Real Yahoo Finance cache | 5 pre-specified economically related pairs | Temporal stability and ML risk-control | `outputs_m7_public/m7/` |
| Exhaustive public walk-forward | Real Yahoo Finance cache | All 45 pairs per fold (135 pair-fold tests) | Multiple-testing robustness | `outputs_exhaustive_public/` |
| Independent trade replay | Same real test data | Same selected signals | Validate primary backtest implementation | `outputs_m7_public/m7/tables/independent_backtest_comparison.csv` |
| Statistical/trading sensitivity | Real public data | Pre-specified sensitivity grid | Threshold/cost/holding-period robustness | `outputs_m7_public/m7/tables/` |
| Pre-specified ETF replication | Real public ETF data only | 8 economic ETF pair hypotheses | External/economic replication on near-substitute instruments | `outputs_m7_etf/` after execution |

## Interpretation rule

A pair is called **statistically validated in a given experiment** only when it passes the correlation, I(1), Engle-Granger, residual ADF, and BH-FDR gates defined for that experiment's pre-specified hypothesis family. Positive out-of-sample PnL alone is not sufficient.
