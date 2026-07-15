# ML-Enhanced Robust Equity Pairs Trading under Synthetic Market Regimes

This repository contains the MScFE 690 capstone research codebase for **ML-enhanced robust equity pairs trading under synthetic market regimes**.

The project investigates whether machine learning and optional deep learning can improve a classical equity/ETF pairs-trading strategy by acting as a **decision filter**. The goal is not to predict raw prices directly, but to decide whether a statistical trade signal should be accepted, rejected, or treated as a possible warning of regime breakdown.

## Current status

This is a **Module 3 / project-proposal-stage codebase**, not the final capstone implementation. It is further developed from the previous Module 2 work and includes runnable code, preliminary outputs, figures, documentation, and archived notes from earlier submissions. The final capstone stage should later replace or supplement these temporary results with final historical-data runs, expanded robustness tests, and final conclusions.

## Research idea

Classical pairs trading can produce false signals when spread deviations are caused by volatility shocks, correlation breaks, liquidity changes, news-driven repricing, or transaction costs. This project combines:

1. A classical cointegration/correlation-based pairs-trading baseline.
2. Feature engineering around spread stability, mean reversion, volatility, correlation, and drawdown.
3. Supervised ML accept/reject trade filters.
4. Synthetic regime generation for robustness testing.
5. Out-of-sample backtesting with transaction costs and slippage assumptions.

## Data frequency

The initial reproducible baseline uses **daily public equity/ETF adjusted close prices**. The code can attempt to use `yfinance`, but the default configuration uses an offline synthetic daily data generator so that instructors can run the project without internet access.

## Repository structure

```text
config/config.yaml                            Main experiment configuration
run_pipeline.py                               Full research pipeline
single_file_demo.py                           Minimal offline demo
src/data.py                                   Data loading and synthetic daily price generation
src/pairs.py                                  Correlation, cointegration, hedge ratio, pair screening
src/features.py                               Spread, z-score, volatility, correlation, drawdown, half-life features
src/labels.py                                 Accept/reject labels based on convergence after costs
src/models.py                                 Logistic regression, random forest, optional XGBoost/fallback boosting filters
src/backtest.py                               Baseline and ML-filtered backtests
src/synthetic.py                              Synthetic market-regime spread scenarios
src/diagnostics.py                            Legacy / optional diagnostics helpers from previous work
src/reporting.py                              Legacy / optional reporting helpers from previous work
src/evaluation.py                             Initial inference text generation
src/visualization.py                          Figures for equity curves, scenario volatility, feature importance
outputs/                                      Generated tables, figures, and findings
docs/MODULE3_CHANGES.md                       Summary of Module 3 code development
docs/RUN_INSTRUCTIONS.md                      Command-line run instructions
docs/CODE_HIGHLIGHTS.md                       Important code sections to highlight in the report
docs/module2_archive/                         Preserved Module 2 notes and templates
scripts/run_smoke_test.py                     Smoke test for reproducibility
tests/test_smoke.py                           Pytest smoke test
CONTRIBUTORS.md                               Contributor list
requirements.txt                              Python dependencies
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the full pipeline

```bash
python run_pipeline.py --config config/config.yaml
```

The outputs are written to:

```text
outputs/tables/
outputs/figures/
outputs/initial_findings.md
```

## Run the minimal demo

```bash
python single_file_demo.py
```

## Run smoke tests

```bash
python scripts/run_smoke_test.py
```

or with pytest:

```bash
python -m pytest tests
```

## Initial results included

The repository includes preliminary outputs produced from reproducible sample data. These outputs demonstrate that the code can:

- screen candidate pairs,
- construct spread features,
- build ML labels,
- train ML filters,
- backtest baseline and ML-filtered strategies,
- generate synthetic stress scenarios,
- create charts with axes and labels,
- export temporary results for the report.

## Important code sections for the report

See `docs/CODE_HIGHLIGHTS.md` for a report-ready list of the most important source-code modules.

## GitHub submission note

For the course submission, upload the contents of this folder to the GitHub repository:

```text
https://github.com/El-Moatasem/robust-pairs-trading-synthetic-regimes
```

Then make sure the professor has read access to the repository.

## Academic note

The included results are **temporary** and should not be interpreted as final investment performance or investment advice. They are included to show project progress, source-code development, and initial inferences for the Module 3 project proposal submission.
