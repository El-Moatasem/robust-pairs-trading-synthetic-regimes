# Final Source Code Guideline Checklist

- [x] README explains the problem, repository, setup, execution, results, and interpretation.
- [x] `requirements.txt` supports `pip install -r requirements.txt`.
- [x] Public verification dataset is included at `data/raw/public_prices.csv`.
- [x] Strict public mode fails instead of silently falling back to synthetic data.
- [x] Pair selection and hedge-ratio estimation use training data only.
- [x] Candidate pairs are canonically oriented so CSV/config order cannot change the result.
- [x] I(1), Engle-Granger, residual ADF, FDR, half-life and pass/fail are reported.
- [x] Predictors are lagged; purge and embargo are applied.
- [x] Model/threshold selection uses validation data only.
- [x] Trading results include costs, drawdown, turnover and confidence intervals.
- [x] Independent trade-replay backtest verifies primary trade accounting.
- [x] Walk-forward analysis repeats pair/model selection through time.
- [x] Statistical, trading-rule, cost and subperiod sensitivity are implemented.
- [x] EDA and final robustness charts include axes, labels and scales.
- [x] Unit tests pass (`9 passed`).
- [x] Smoke test passes.
- [x] `.venv`, `local_directory`, `.git`, caches and IDE files are excluded from the submission ZIP.
