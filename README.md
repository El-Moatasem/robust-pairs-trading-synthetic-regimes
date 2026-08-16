# ML-Enhanced Robust Equity Pairs Trading under Synthetic Market Regimes

Final MScFE 690 capstone implementation by **El Moatasem Madani and Henry A. Sada** (Group 16021).

The research asks whether machine learning and deep learning can improve a classical cointegration-based pairs-trading strategy by acting as a **signal-quality filter** rather than a raw-price forecaster. The repository contains deterministic synthetic experiments, genuine public-market experiments, leakage controls, independent backtest validation, walk-forward analysis, sensitivity analysis, deep neural-network and GRU comparisons, and report-ready exploratory diagnostics.

## Final research position

The final code does **not** claim a universally profitable ML/DL strategy. The final extension adds a compact deep MLP and a 20-day sequence-aware PyTorch GRU while keeping the classical ML models as unchanged benchmarks. The evidence is conditional:

- The fixed public-data experiment selects **CVX-XOM provisionally** from the full 10-asset universe. The validation-selected random forest improves observed test-period PnL, Sharpe, drawdown, turnover, and costs versus the classical baseline, but CVX-XOM fails the full-sample Engle-Granger/FDR screen and the economic confidence intervals include zero.
- In the M7 economically pre-specified five-pair walk-forward analysis, **CVX-XOM passes all configured screens in folds 1 and 2**. Fold 2 (test: 2023-01-06 to 2024-01-04) is profitable out of sample with net log-spread PnL **0.1007** and Sharpe **0.9014**, but the validation-selected ML filter accepts the same trades as the baseline in that fold. This is evidence that the pair traded profitably in that historical window, **not** evidence that ML created the profit or that future profit is guaranteed.
- Fold 1 passes the five-hypothesis screen. The original validation-selected ML result was not profitable, but the expanded DL comparison finds positive OOS results for random forest and the fixed deep MLP in that fold. Fold 3 does not pass the full pair screen.

- In the DL-extended five-pair walk-forward experiment, **CVX-XOM is the only pair that passes every configured screen, in folds 1 and 2**. Random forest and the fixed deep MLP both generate positive OOS PnL in fold 1; every tested ML/DL filter is positive in fold 2. The validation-selected cross-model winner is still not universally profitable across all folds.
- A strict statistical-gate policy (stay flat whenever no pair passes every screen) is non-negative across all three observed folds for the fixed random forest and fixed deep MLP. The deep MLP produces aggregate gated PnL **0.1110** across the three folds (positive in both valid-pair folds and no trade in fold 3). This is an encouraging observed stability result, **not proof of universal future profitability**.
- An additional **exhaustive 45-pair walk-forward screen** tests every unordered combination of the original 10 real tickers in each of three folds (135 pair-fold hypotheses). No case survives the BH-FDR gate when correction is applied across all 45 pairs in a fold. Some raw-diagnostic cases have positive OOS PnL, but these are explicitly labeled exploratory because they fail the global multiple-testing gate.
- A final **economically pre-specified ETF replication** is encoded separately using SPY-IVV, SPY-VOO, IVV-VOO, QQQ-QQQM, GLD-IAU, IWM-VTWO, VTI-ITOT, and AGG-BND. It runs only on real public data in strict mode and never substitutes synthetic data.
- The independent trade-replay backtest reproduces the primary engine's trade counts and PnL to numerical precision.
- Synthetic regimes show no universal ML advantage; the ML filter is most useful as a robustness/risk-control research layer rather than a guaranteed alpha source.

## Repository structure

```text
config/config.yaml                 Deterministic synthetic-development run
config/config_public.yaml          Full 10-asset public-data discovery run
config/config_m7_public.yaml       M7 economic-pair + walk-forward/sensitivity run
config/config_m7_etf.yaml          Pre-specified strict real-ETF replication (requires download/cache)
run_pipeline.py                    Main end-to-end pipeline
single_file_demo.py                Minimal offline demonstration
src/data.py                        Public data + deterministic synthetic generator
src/splits.py                      Purged chronological train/validation/test splits
src/pairs.py                       I(1), EG, residual ADF, FDR, half-life screening
src/features.py                    Lagged feature engineering
src/labels.py                      Cost-aware convergence labels
src/models.py                      Classical ML + deep MLP validation-only selection
src/deep_learning.py               Sequence-aware GRU, DL walk-forward, universality/gating tests
src/backtest.py                    Primary test-period backtester
src/independent_backtest.py        Independent trade-replay verification engine
src/walkforward.py                 Expanding-window walk-forward analysis
src/sensitivity.py                 Statistical/trading/subperiod sensitivity
src/eda.py                         Exploratory charts requested in final feedback
src/m7_visualization.py            Final walk-forward/sensitivity figures
src/synthetic.py                   Calibrated synthetic regime analysis
scripts/run_m7_analysis.py         M7/final robustness orchestrator
scripts/run_deep_learning_analysis.py  Deep MLP + GRU fixed-split/walk-forward experiments
scripts/summarize_deep_results.py  Passed/profitable and universality summary
scripts/run_exhaustive_public_walkforward.py  All 45 original real pairs across walk-forward folds
scripts/summarize_etf_results.py   Summarize strict real-ETF replication
scripts/run_all_final_experiments.py  Convenience runner for final experiment suite
scripts/run_smoke_test.py          End-to-end smoke test
scripts/verify_outputs.py          Output/research-control verifier
tests/test_research_controls.py    Research-control and reproducibility tests
data/raw/public_prices.csv         Included verification dataset for reported public results
outputs/                            Synthetic results
outputs_public/                     Full public-discovery results
outputs_m7_public/                  M7 five-pair robustness outputs + EDA/walk-forward results
outputs_exhaustive_public/          45-pair real-data walk-forward screening/results
outputs_deep_synthetic/              Synthetic deep-learning comparison
outputs_deep_public/                 Full-public fixed-split deep-learning comparison
outputs_deep_m7_public/              ML/DL walk-forward, strict-gate, universality results
docs/                               Methodology, run guide, solution design, final findings
```

## Setup

### macOS / Linux

```bash
cd robust-pairs-trading-synthetic-regimes
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
cd robust-pairs-trading-synthetic-regimes
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 1. Run deterministic synthetic data

```bash
python run_pipeline.py --config config/config.yaml
python scripts/verify_outputs.py --outputs outputs
```

Expected verification includes:

```text
Output verification passed.
Synthetic data: True
```

## 2. Run the full genuine public-data experiment

The repository includes the exact public adjusted-close cache used for the reported results at `data/raw/public_prices.csv`.

```bash
python run_pipeline.py --config config/config_public.yaml --require-public-data
python scripts/verify_outputs.py --outputs outputs_public
```

Expected verification includes:

```text
Output verification passed.
Synthetic data: False
```

To force a fresh Yahoo Finance download instead of the included cache:

```bash
rm -f data/raw/public_prices.csv
python run_pipeline.py --config config/config_public.yaml --require-public-data
```

## 3. Run the M7/final robustness analysis

First generate the M7 fixed-split outputs:

```bash
python run_pipeline.py --config config/config_m7_public.yaml --require-public-data
python scripts/verify_outputs.py --outputs outputs_m7_public
```

Then run the final extensions:

```bash
python scripts/run_m7_analysis.py \
  --config config/config_m7_public.yaml \
  --require-public-data
```

This produces:

- EDA appendix figures;
- independent backtest agreement tables;
- 10% vs 5% significance sensitivity;
- trading-rule/cost sensitivity;
- public-test subperiod stability;
- expanding-window walk-forward pair selection, model selection, and OOS backtesting;
- a table of walk-forward cases that both pass the configured screens and produce positive OOS PnL.

Key file:

```text
outputs_m7_public/m7/tables/passed_and_profitable_walk_forward.csv
```

## 4. Exhaustive all-45 public-pair walk-forward

This experiment removes the five-pair restriction and screens all 45 unordered pairs formed by the original 10 real tickers in every walk-forward formation window. BH-FDR is applied across all 45 hypotheses within each fold.

```bash
python scripts/run_exhaustive_public_walkforward.py \
  --config config/config_m7_public.yaml \
  --require-public-data
```

The included verified screening output contains 135 pair-fold cases (45 pairs x 3 folds) and zero full 45-pair FDR passes. The slower exploratory economic pass can be reproduced with `--backtest-raw`; positive PnL in that table must not be interpreted as statistical validation when the global FDR gate fails.

## 5. Deep-learning extension

Run the fixed-split synthetic and public neural comparisons:

```bash
python scripts/run_deep_learning_analysis.py --config config/config.yaml --output-dir outputs_deep_synthetic
python scripts/run_deep_learning_analysis.py --config config/config_public.yaml --require-public-data --output-dir outputs_deep_public
```

Run the full ML/DL walk-forward comparison on the economically pre-specified real pair family:

```bash
python scripts/run_deep_learning_analysis.py \
  --config config/config_m7_public.yaml \
  --require-public-data \
  --walk-forward \
  --output-dir outputs_deep_m7_public
python scripts/summarize_deep_results.py --outputs outputs_deep_m7_public
```

Key outputs include `deep_passed_and_profitable.csv`, `deep_validation_selected_walk_forward.csv`, `strict_statistical_gate_model_summary.csv`, and `deep_universality_summary.json`.

## 6. Economically pre-specified real-ETF replication

`config/config_m7_etf.yaml` tests economically close ETF pairs such as SPY-IVV, SPY-VOO, QQQ-QQQM, GLD-IAU, VTI-ITOT and AGG-BND. These hypotheses are specified by economic exposure before test-period profitability is inspected. The final package does **not** contain fabricated ETF results: if the real ETF cache is absent and Yahoo Finance is unavailable, strict mode stops. Numerical ETF findings should be added to the report only after a verified `Synthetic data: False` run.

```bash
rm -f data/raw/etf_prices.csv
python run_pipeline.py --config config/config_m7_etf.yaml --require-public-data
python scripts/verify_outputs.py --outputs outputs_m7_etf
python scripts/run_m7_analysis.py --config config/config_m7_etf.yaml --require-public-data
python scripts/summarize_etf_results.py --outputs outputs_m7_etf
```

## 7. Run the final suite

All reproducible experiments backed by included data can be run with:

```bash
python scripts/run_all_final_experiments.py
```

In a network-enabled environment, add the strict ETF replication:

```bash
python scripts/run_all_final_experiments.py --with-etf
```

## 8. Tests and smoke test

```bash
python -m pytest -q
python scripts/run_smoke_test.py
```

Expected unit-test result for this final package:

```text
13 passed
```

The tests cover pair-screening controls, candidate-pair restriction, leakage safeguards, independent backtest agreement, and invariance of pair screening to cached CSV column order.

## Important reproducibility control: deterministic pair orientation

Finite-sample Engle-Granger/OLS diagnostics can change when the dependent and independent tickers are swapped. The final code uses a **canonical alphabetical pair orientation**, so YAML ticker order or cached CSV column order cannot silently change the research result. A unit test explicitly verifies this behavior.

## Interpretation of the profitable CVX-XOM walk-forward fold

Within the economically pre-specified five-pair walk-forward experiment, the only selected fold that both passed that experiment's configured screen and had positive selected-strategy OOS PnL is:

```text
Fold:              2
Pair:              CVX-XOM
Test period:       2023-01-06 to 2024-01-04
Engle-Granger p:   0.01063
Residual ADF p:    0.000118
BH-FDR p:          0.05315
Estimated half-life: 16.97 trading days
Net log-spread PnL:  0.10072
Sharpe:              0.90139
Closed trades:       12
Win rate:             58.33%
```

This is a historical out-of-sample research result, **not a recommendation to trade CVX-XOM and not a guarantee of future returns**. In that fold the ML filter and classical baseline produced the same economic path, so the profit should be attributed to the pair/trading rule rather than to incremental ML filtering.

## GitHub

Repository used in the capstone report:

`https://github.com/El-Moatasem/robust-pairs-trading-synthetic-regimes`

## Final advanced robustness extension

The final package also evaluates four additional robustness techniques without replacing the original ML/DL evidence:

- **LSTM** sequence classifier.
- **Causal TCN** (dilated 1-D temporal convolutions).
- **Regime mixture-of-experts**, where regime thresholds are estimated from formation data and experts are selected by validation F1 only.
- **Uncertainty consensus / abstention**, where the minimum number of agreeing experts is selected on validation data and ambiguous signals are rejected.

Run the genuine public walk-forward extension with:

```bash
python scripts/run_advanced_learning_analysis.py \
  --config config/config_m7_public.yaml \
  --require-public-data \
  --walk-forward \
  --output-dir outputs_advanced_m7_public
python scripts/summarize_advanced_results.py --outputs outputs_advanced_m7_public
```

The strongest single advanced test-period result is the fold-2 CVX-XOM **regime mixture-of-experts** result (PnL about `0.16382`, Sharpe about `1.5899`). It does not remain profitable in fold 1. Under the strict statistical gate, the most stable observed fixed policies remain **random forest** (aggregate gated PnL about `0.14851`) and **deep MLP** (about `0.11103`), each profitable in both statistically valid CVX-XOM folds and flat in the rejected third fold.

No universal ML/DL strategy is claimed. The package explicitly distinguishes a strict universal-profitability test from the weaker observed-gated-robustness result.
