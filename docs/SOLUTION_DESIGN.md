# Solution Design - ML-Enhanced Robust Equity Pairs Trading

## Business problem
Classical pairs trading can confuse a large spread deviation with a genuine mean-reversion opportunity. Pair relationships can decay, multiple testing can create false discoveries, trading costs can remove apparent profits, and ML/backtests can leak future information.

## Solution architecture
1. **Data layer (`src/data.py`)** - loads deterministic synthetic data or strict real public adjusted-close data. Strict real mode stops rather than falling back to synthetic.
2. **Time controls (`src/splits.py`, `src/walkforward.py`)** - chronological formation/validation/test splits with purge and embargo; walk-forward re-estimation repeats the full research process.
3. **Pair-quality layer (`src/pairs.py`)** - deterministic ticker orientation, correlation, I(1), Engle-Granger, residual ADF, half-life, and BH-FDR.
4. **ML decision layer (`src/features.py`, `src/labels.py`, `src/models.py`)** - one-day-lagged relationship-quality features, cost-aware labels, and validation-only model/threshold selection.
5. **Trading/verification (`src/backtest.py`, `src/independent_backtest.py`)** - primary cost-aware backtest plus independent trade replay.
6. **Final robustness (`src/sensitivity.py`, `src/eda.py`, `scripts/run_m7_analysis.py`)** - EDA, significance/trading sensitivity, subperiod analysis, and walk-forward evaluation.
7. **Multiple-testing audit (`scripts/run_exhaustive_public_walkforward.py`)** - all 45 real pairs from the original 10-ticker universe are screened in every fold; BH-FDR is applied across the full 45-hypothesis family.
8. **External ETF replication (`config/config_m7_etf.yaml`)** - eight economically pre-specified real-ETF pair hypotheses are evaluated using the same strict controls. Numerical ETF results are only valid after `Synthetic data: False` verification.

## Key verified results
- The fixed public CVX-XOM experiment is provisional, but the selected RF improves the observed risk/return path.
- In the five-pair walk-forward family, CVX-XOM fold 2 passes the 10% family-level screen and earns OOS PnL 0.1007 / Sharpe 0.9014; ML does not add incremental PnL in that fold.
- In the exhaustive 45-pair-per-fold robustness screen (135 pair-fold tests), **zero pairs survive the global BH-FDR gate**. Positive raw-diagnostic OOS cases are therefore exploratory rather than statistically validated.
- The independent replay engine matches primary trade counts and PnL to numerical precision.

## Deployment decision
Use the package as a research/risk-control framework, not an automatic trade recommendation. Production deployment would require intraday execution data, borrow/financing constraints, market impact, monitoring, and portfolio-level risk controls.

### Advanced decision layer
After the statistical pair gate, the final package can compare a fixed RF/MLP/GRU/LSTM/TCN expert set. A formation-window regime classifier can route a signal to a validation-selected expert, while an uncertainty-consensus policy can abstain when too few experts agree. Neither policy is allowed to weaken the statistical pair screen, and neither uses test PnL for tuning.
