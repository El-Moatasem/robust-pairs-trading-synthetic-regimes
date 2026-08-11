# Final Run and Verification Instructions

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Synthetic analysis

```bash
python run_pipeline.py --config config/config.yaml
python scripts/verify_outputs.py --outputs outputs
```

## Full public-data analysis

```bash
python run_pipeline.py --config config/config_public.yaml --require-public-data
python scripts/verify_outputs.py --outputs outputs_public
```

The included `data/raw/public_prices.csv` allows the instructor to reproduce the reported public results without relying on a future Yahoo Finance download. Delete that cache only when a fresh download is intentionally desired.

## M7/final robustness analysis

```bash
python run_pipeline.py --config config/config_m7_public.yaml --require-public-data
python scripts/verify_outputs.py --outputs outputs_m7_public
python scripts/run_m7_analysis.py --config config/config_m7_public.yaml --require-public-data
```

Inspect:

```text
outputs_m7_public/m7/tables/walk_forward_summary.csv
outputs_m7_public/m7/tables/walk_forward_pair_screens.csv
outputs_m7_public/m7/tables/passed_and_profitable_walk_forward.csv
outputs_m7_public/m7/tables/independent_backtest_comparison.csv
outputs_m7_public/m7/tables/trading_rule_sensitivity.csv
outputs_m7_public/m7/tables/subperiod_stability.csv
outputs_m7_public/m7/figures/
```

## Exhaustive all-45 public-pair walk-forward

```bash
python scripts/run_exhaustive_public_walkforward.py --config config/config_m7_public.yaml --require-public-data
```

To reproduce the slower exploratory OOS backtests for raw-diagnostic candidates that do not necessarily survive the global 45-pair FDR gate:

```bash
python scripts/run_exhaustive_public_walkforward.py --config config/config_m7_public.yaml --require-public-data --backtest-raw
```

## Economically pre-specified related-ETF replication

Requires internet access or a matching `data/raw/etf_prices.csv`:

```bash
python run_pipeline.py --config config/config_m7_etf.yaml --require-public-data
python scripts/verify_outputs.py --outputs outputs_m7_etf
python scripts/run_m7_analysis.py --config config/config_m7_etf.yaml --require-public-data
python scripts/summarize_etf_results.py --outputs outputs_m7_etf
```

## Tests

```bash
python -m pytest -q
```

Expected:

```text
9 passed
```

## Smoke test

```bash
python scripts/run_smoke_test.py
```

Expected final line:

```text
Smoke test passed.
```
