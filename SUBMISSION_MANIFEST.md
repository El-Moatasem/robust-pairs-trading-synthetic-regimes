# Final Submission Manifest - Group 16021

Submit the final project report PDF separately from the compressed source-code package, plus the one-page Solution Design Document as required by the course instructions.

## Source package contents

- `README.md` - setup, experiment matrix, commands, interpretation.
- `requirements.txt` - Python dependencies.
- `config/config.yaml` - deterministic synthetic experiment.
- `config/config_public.yaml` - original real 10-ticker / 45-pair fixed-split experiment.
- `config/config_m7_public.yaml` - economically pre-specified five-pair M7 walk-forward experiment.
- `config/config_m7_etf.yaml` - eight-hypothesis strict real-ETF replication.
- `scripts/run_m7_analysis.py` - EDA, independent replay, walk-forward and sensitivity suite.
- `scripts/run_deep_learning_analysis.py` - deep MLP + GRU fixed-split and walk-forward experiments.
- `scripts/summarize_deep_results.py` - passed/profitable, strict-gate, and universality summaries.
- `scripts/run_exhaustive_public_walkforward.py` - all 45 original public pairs in each walk-forward fold.
- `scripts/summarize_etf_results.py` - report-ready ETF result summary after a verified real-data run.
- `scripts/run_all_final_experiments.py` - convenience runner.
- `outputs_exhaustive_public/` - verified 135-row all-45 screening results and exploratory raw-diagnostic backtest snapshot.
- `outputs_deep_synthetic/`, `outputs_deep_public/`, `outputs_deep_m7_public/` - executed neural-model results.
- `data/raw/public_prices.csv` - included genuine public-data cache used for the reported 10-ticker results.
- `data/ETF_DATA_README.md` - strict ETF data/reproduction instructions; no synthetic ETF cache is substituted.

## Excluded local artifacts

Do not submit `.venv/`, `.git/`, `local_directory/`, `__pycache__/`, `.pytest_cache/`, IDE metadata, or other machine-local caches.

## Verification

```bash
python -m pytest -q
# Expected: 13 passed
python scripts/verify_outputs.py --outputs outputs
python scripts/verify_outputs.py --outputs outputs_public
python scripts/verify_outputs.py --outputs outputs_m7_public
```

### Advanced robustness additions
- `src/advanced_learning.py`
- `scripts/run_advanced_learning_analysis.py`
- `scripts/summarize_advanced_results.py`
- `docs/ADVANCED_LEARNING_RESULTS.md`
- `outputs_advanced_synthetic/`
- `outputs_advanced_public/`
- `outputs_advanced_m7_public/`
- `FINAL_ADVANCED_RESULT_SUMMARY.txt`
