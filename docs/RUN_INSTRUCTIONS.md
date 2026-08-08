# Detailed Run and Verification Instructions

## 1. Open a terminal in the repository

```bash
cd robust-pairs-trading-synthetic-regimes_FINAL
```

## 2. Create an isolated Python environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

The shell prompt should show `(.venv)`.

## 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Do not install into Homebrew's system-managed Python with `--break-system-packages`.

## 4. Run the reproducible offline pipeline

```bash
python run_pipeline.py --config config/config.yaml
```

## 5. Run the final public-data analysis

```bash
python run_pipeline.py --config config/config_public.yaml --require-public-data
```

If the computer cannot access Yahoo Finance, place a CSV at `data/raw/public_prices.csv`. The first column must contain dates and the remaining columns must contain adjusted-close series with ticker names as headers.

## 6. Run the tests

```bash
python -m pytest -q
```

Expected result:

```text
4 passed
```

## 7. Run the smoke test

```bash
python scripts/run_smoke_test.py
```

Expected final line:

```text
Smoke test passed.
```

## 8. Verify outputs automatically

```bash
python scripts/verify_outputs.py --outputs outputs
```

Expected final line:

```text
Output verification passed.
```

## 9. Manual verification checklist

Open the following files:

- `outputs/tables/data_summary.json`: confirm whether the run is synthetic or public.
- `outputs/tables/time_split_summary.json`: confirm train, validation, test, purge, and embargo information.
- `outputs/tables/candidate_pairs.csv`: confirm the selected pair and diagnostic pass flags.
- `outputs/tables/model_metrics.csv`: confirm model selection used validation metrics and inspect test metrics separately.
- `outputs/tables/strategy_metrics_test.csv`: compare the untouched test-period baseline and all ML-filtered variants.
- `outputs/tables/synthetic_regime_summary.csv`: inspect performance by regime.
- `outputs/run_summary.json`: confirm the selected pair, model, data source, and important caveat.

## 10. PyCharm

Set the interpreter to:

```text
<project>/.venv/bin/python
```

Create a run configuration with:

```text
Script path: run_pipeline.py
Parameters: --config config/config.yaml
Working directory: repository root
```
