# ML-Enhanced Robust Equity Pairs Trading under Synthetic Market Regimes

This repository contains the revised MScFE 690 capstone research pipeline developed by **El Moatasem Madani and Henry A. Sada**.

The project tests whether machine learning can act as a **trade-decision filter** for a classical cointegration-based equity/ETF pairs-trading strategy. It does not use ML to predict raw prices. Instead, the models estimate whether a statistically identified spread deviation is likely to converge economically after costs and before a stop-loss or maximum holding horizon.

## Current research status

This codebase incorporates the peer-review feedback received after the project-proposal stage. It is a corrected and reproducible research implementation, but the default run remains an **offline synthetic-data development run**. It should not be presented as evidence of historical profitability.

For the final empirical analysis, run `config/config_public.yaml` with internet access or provide a clean adjusted-close CSV at `data/raw/public_prices.csv`.

## What was fixed

The revised implementation now:

- labels synthetic and public data explicitly and never silently presents synthetic ticker labels as historical evidence;
- performs pair screening and hedge-ratio estimation only in the training window;
- checks that both price series are plausibly I(1);
- applies Engle-Granger, a matching residual ADF diagnostic, and Benjamini-Hochberg false-discovery control;
- reports a pair as provisional if conservative diagnostics are not all satisfied;
- freezes the cointegrating intercept and hedge ratio before validation and testing;
- shifts all model predictors by one trading day;
- defines the z-score, half-life, deviation persistence, and regime-stress proxy explicitly;
- creates labels only at valid trade-entry signals and includes four-leg round-trip pair costs;
- uses purged and embargoed chronological train/validation/test partitions;
- selects the model and probability threshold using validation data only;
- backtests every candidate model on the same untouched test period;
- separates classification metrics from economic trading metrics;
- reports block-bootstrap confidence intervals for test-period cumulative PnL and Sharpe ratio;
- calibrates synthetic OU-style spread regimes from the training spread;
- tests calm, high-volatility, jump, weak-mean-reversion, and stress regimes;
- includes tests confirming no future-data changes alter past predictors.


## M6 draft-project submission status

This repository is the **Module 6 draft-project source-code snapshot**. It contains the peer-review revisions, reproducible current results, and the documented plan for the remaining empirical work.

For M6, the default configuration intentionally uses deterministic synthetic development data so the instructor can reproduce the complete workflow offline. The resulting figures and metrics are **methodology-development evidence**, not claims of historical profitability. The strict public-data configuration is included as the next empirical stage and fails explicitly if genuine public data are unavailable.

The M6 rubric asks for code that has been further developed since the previous submission, appropriate code comments, and a detailed discussion of results and their significance. The relevant materials are:

- `docs/PEER_REVIEW_RESPONSE.md` - issue-by-issue revisions made after peer review.
- `docs/METHODOLOGY.md` - equations, assumptions, labels, splits, and backtest definitions.
- `docs/RESULTS_AND_SIGNIFICANCE.md` - current results and the inferences that can and cannot be drawn.
- `docs/M6_PROGRESS_AND_SCHEDULE.md` - completed work, remaining work, and schedule.
- `docs/SOURCE_CODE_GUIDELINE_CHECKLIST.md` - mapping to WQU source-code requirements.
- `M6_SUBMISSION.md` - submission/run checklist.

## Repository structure

```text
config/config.yaml                 Reproducible offline synthetic development run
config/config_public.yaml          Strict public-data run; fails if data cannot be loaded
run_pipeline.py                    End-to-end revised research pipeline
single_file_demo.py                Minimal offline demonstration
src/data.py                        Public-data loading and cointegrated synthetic generator
src/splits.py                      Purged chronological train/validation/test split
src/pairs.py                       I(1), correlation, Engle-Granger, residual ADF, FDR screening
src/features.py                    Lagged feature engineering without future information
src/labels.py                      Cost-aware convergence labels at valid entry signals
src/models.py                      Validation-only model and threshold selection
src/backtest.py                    Test-period baseline and ML-filtered backtests
src/synthetic.py                   Calibrated synthetic regime generation and evaluation
src/visualization.py               Report-ready figures
src/evaluation.py                  Reproducible findings summary
scripts/run_smoke_test.py          End-to-end smoke test
scripts/verify_outputs.py          Checks required outputs and key research controls
tests/test_research_controls.py    Unit tests for cointegration, leakage, labels, and splits
docs/METHODOLOGY.md                Mathematical definitions and assumptions
docs/PEER_REVIEW_RESPONSE.md       Peer-review issue-to-fix matrix
docs/RUN_INSTRUCTIONS.md           Detailed setup and run guide
docs/module2_archive/              Preserved earlier work
docs/module3_archive/              Preserved earlier code documentation
outputs/                            Current reproducible synthetic-development results
```

## Setup on macOS or Linux

```bash
cd robust-pairs-trading-synthetic-regimes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the offline reproducible pipeline

```bash
python run_pipeline.py --config config/config.yaml
```

## Run with public adjusted-close data

```bash
python run_pipeline.py --config config/config_public.yaml --require-public-data
```

Public mode intentionally fails if data cannot be downloaded or loaded from the configured cache. This prevents an unnoticed fallback from being reported as a historical-data result.

## Run tests and verification

```bash
python -m pytest -q
python scripts/run_smoke_test.py
python scripts/verify_outputs.py --outputs outputs
```

Expected unit-test result:

```text
4 passed
```

Expected pipeline completion messages include:

```text
Pipeline completed successfully.
Pair-selection window ends: ...
Selected model (validation only): ...
Test window: ...
```

## Main outputs

```text
outputs/run_summary.json
outputs/initial_findings.md
outputs/config_used.yaml
outputs/tables/data_summary.json
outputs/tables/time_split_summary.json
outputs/tables/candidate_pairs.csv
outputs/tables/feature_definitions.csv
outputs/tables/labels.csv
outputs/tables/model_metrics.csv
outputs/tables/strategy_metrics_test.csv
outputs/tables/synthetic_calibration.json
outputs/tables/synthetic_regime_summary.csv
outputs/figures/pair_spread_and_signal.png
outputs/figures/out_of_sample_equity_comparison.png
outputs/figures/out_of_sample_roc_curves.png
outputs/figures/selected_model_feature_importance.png
outputs/figures/synthetic_regime_performance.png
```

## Current reproducible synthetic-development finding

The current default run selected XOM-CVX from a deliberately cointegrated synthetic universe using the training window only. The conservative diagnostics agreed in this revised generator. Random forest was selected using validation F1, but its test AUC was below 0.5, which indicates poor out-of-sample generalization. A logistic-regression filter produced the strongest test-period Sharpe among the reported variants, but it was not the model selected in advance by validation. Therefore, the current result does **not** establish that ML robustly improves the strategy. It demonstrates a corrected evaluation workflow and identifies the next empirical questions for the public-data run.

## Reproducibility note

The default synthetic generator creates designated pairs using a shared I(1) stochastic trend plus a stationary AR(1) residual. The synthetic results are useful for software verification, leakage testing, and scenario analysis. They are not substitutes for the final public equity/ETF analysis.
