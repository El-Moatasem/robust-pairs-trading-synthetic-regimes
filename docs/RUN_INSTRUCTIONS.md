# Run Instructions

## Create environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run quick demo

```bash
python single_file_demo.py
```

## Run full pipeline

```bash
python run_pipeline.py --config config/config.yaml
```

## Run smoke test

```bash
python scripts/run_smoke_test.py
```

## Outputs

Generated outputs are written to:

```text
outputs/tables/
outputs/figures/
outputs/initial_findings.md
```
