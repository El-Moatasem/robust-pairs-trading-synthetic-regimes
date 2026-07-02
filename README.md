# MScFE 690: ML-Enhanced Robust Equity Pairs Trading under Synthetic Market Regimes

This repository contains the capstone research codebase for **ML-Enhanced Robust Equity Pairs Trading under Synthetic Market Regimes** under **Track 8 - Machine Learning (Deep) Investment Strategies**.

## Project summary

Classical equity pairs trading assumes that two economically related securities maintain a stable spread and that temporary deviations will mean-revert. In practice, many apparent spread deviations are caused by volatility shocks, correlation breakdowns, liquidity changes, regime shifts, news-driven repricing, or transaction costs. This project studies whether machine learning and optional deep learning can improve a classical pairs-trading strategy by acting as a **decision layer** that accepts, rejects, or flags trade signals.

The project combines:

- Classical statistical arbitrage and pairs trading.
- Cointegration and spread-based baseline modeling.
- ML trade filtering using logistic regression, random forest, and optional XGBoost.
- Optional DL/regime modules such as LSTM, GRU, or autoencoder-based anomaly detection.
- Synthetic market scenarios for robustness testing under volatility, jump, correlation-breakdown, mean-reversion, slippage, and transaction-cost regimes.
- Model, trading, risk, robustness, and interpretability metrics.

## Repository description for GitHub

MScFE 690 capstone research codebase for ML-enhanced equity pairs trading, synthetic market-regime robustness testing, cointegration-based statistical arbitrage, ML trade filtering, and risk-aware backtesting.

## Repository structure

```text
.
├── config/
│   └── config.yaml                  # Main experiment configuration
├── data/
│   ├── raw/                         # Raw downloaded or user-provided data
│   └── processed/                   # Cleaned feature datasets
├── docs/
│   ├── initial_findings_template.md # Temporary-results report template
│   └── literature_competitor_notes.md
├── notebooks/
│   └── README.md                    # Notebook guidance
├── outputs/
│   ├── figures/                     # Generated charts
│   └── tables/                      # Generated CSV summaries
├── scripts/
│   └── run_smoke_test.py            # Quick verification script
├── src/
│   ├── data.py                      # Data download/load/cleaning
│   ├── diagnostics.py               # ADF, cointegration, residual tests
│   ├── pairs.py                     # Pair screening and ranking
│   ├── features.py                  # Hedge ratio, spread, z-score, half-life features
│   ├── labels.py                    # Trade label construction
│   ├── backtest.py                  # Baseline and ML-filtered strategy backtests
│   ├── models.py                    # ML trade-filter models
│   ├── synthetic.py                 # Synthetic regime generation
│   ├── evaluation.py                # Model/trading/risk metrics
│   ├── reporting.py                 # Tables, summaries, and export helpers
│   └── visualization.py             # Figure creation helpers
├── tests/
│   └── test_smoke.py                # Lightweight unit tests
├── run_pipeline.py                  # End-to-end research pipeline
├── single_file_demo.py              # Self-contained demo for copy/paste testing
├── requirements.txt
└── README.md
```

## Important code sections to highlight in the report

| File | Why it matters |
|---|---|
| `src/pairs.py` | Screens candidate equity/ETF pairs using correlation and cointegration diagnostics. |
| `src/features.py` | Constructs spreads, hedge ratios, z-scores, rolling volatility/correlation, drawdown, and half-life features. |
| `src/labels.py` | Converts historical trading signals into ML targets such as accept/reject or probability of convergence. |
| `src/models.py` | Trains logistic regression, random forest, and optional XGBoost trade-filter models. |
| `src/backtest.py` | Compares classical baseline, risk-aware baseline, and ML-filtered strategy variants after transaction costs. |
| `src/synthetic.py` | Generates synthetic spread regimes for robustness testing under volatility, jumps, correlation breakdowns, mean reversion, slippage, and costs. |
| `run_pipeline.py` | Runs the full research workflow and exports initial findings. |

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run a smoke test

```bash
python scripts/run_smoke_test.py
```

This uses generated sample data and verifies that the main modules work without requiring internet access.

## Run the full pipeline

```bash
python run_pipeline.py --config config/config.yaml
```

The pipeline will:

1. Load public price data if available, or generate a synthetic sample if `use_sample_data: true` is enabled.
2. Screen candidate pairs.
3. Build spread and regime features.
4. Generate baseline pairs-trading signals.
5. Build ML labels.
6. Train ML trade filters.
7. Run baseline and ML-filtered backtests.
8. Generate synthetic market-regime scenarios.
9. Export tables and figures into `outputs/`.

## Configuration

Edit `config/config.yaml` to change:

- Tickers and date range.
- Pair-screening thresholds.
- Entry/exit z-score thresholds.
- Transaction costs.
- ML model selection.
- Synthetic-regime parameters.

## Current status

This repository is an **initial capstone research implementation**. It is not a production trading system. The current focus is to support the literature review, competitor analysis, initial findings, and reproducible research workflow. Future milestones include expanding robustness tests, improving model validation, adding additional metrics, and optionally adding a deeper sequential model.


## Academic note

The repository supports a research workflow only. It does not provide investment advice, live trading recommendations, or a production-ready trading platform.
