# Advanced ML/DL Robustness Extension

This final extension evaluates four pre-declared additions to the original ML/DL framework:

1. **LSTM** sequence classifier using the same 20-day lagged relationship-quality feature window as the GRU.
2. **Causal temporal CNN (TCN)** with two dilated one-dimensional convolution blocks.
3. **Training-defined regime mixture-of-experts (MoE)**. Regime thresholds are estimated from the formation window only; the expert for each regime is selected by validation F1 only.
4. **Uncertainty-aware consensus / abstention policy**. A signal is accepted only when a validation-selected minimum number of fixed experts agree; disagreement implies no trade.

The candidate expert set is fixed before test evaluation: random forest, deep MLP, GRU, LSTM and TCN. All features remain one-day lagged. Test PnL is never used to choose architectures, regime experts, consensus strength or thresholds.

## Fixed-split checks

### Deterministic synthetic data

CVX-XOM passes the complete statistical screen.

| Model | Validation F1 | OOS PnL | Sharpe | Passed + profitable |
|---|---:|---:|---:|---|
| LSTM | 0.6852 | -0.04234 | -0.3409 | No |
| TCN | 0.7143 | **+0.03443** | **+0.2707** | **Yes** |

The TCN is therefore a profitable passed synthetic case, but synthetic profitability is a software/robustness result rather than historical investment evidence.

### Genuine public fixed split

CVX-XOM remains **provisional** in the full 45-pair discovery family, so positive PnL cannot be called statistically validated under that experiment.

| Model | OOS PnL | Sharpe | Pair passes full screen |
|---|---:|---:|---|
| LSTM | +0.00583 | +0.0213 | No |
| TCN | +0.04651 | +0.1887 | No |

## Real-data five-pair walk-forward extension

The statistically valid real pair remains **CVX-XOM**, passing the complete five-pair screen in folds 1 and 2. No candidate passes in fold 3.

### Passed-and-profitable advanced cases

| Fold | Pair | Policy/model | OOS PnL | Sharpe |
|---:|---|---|---:|---:|
| 1 | CVX-XOM | Random forest | +0.04778 | +0.7340 |
| 1 | CVX-XOM | Deep MLP | +0.01031 | +0.1368 |
| 1 | CVX-XOM | Uncertainty consensus | +0.02722 | +0.4353 |
| 2 | CVX-XOM | Random forest | +0.10072 | +0.9014 |
| 2 | CVX-XOM | Deep MLP | +0.10072 | +0.9014 |
| 2 | CVX-XOM | GRU | +0.10072 | +0.9014 |
| 2 | CVX-XOM | LSTM | +0.10072 | +0.9014 |
| 2 | CVX-XOM | TCN | +0.03024 | +0.3798 |
| 2 | CVX-XOM | Regime mixture-of-experts | **+0.12337** | **+1.1642** |
| 2 | CVX-XOM | Uncertainty consensus | +0.10072 | +0.9014 |

The fold-2 regime-aware MoE is the strongest single advanced-model OOS result. Its validation-only regime experts are selected from the fixed candidate set using training-defined regimes. It does **not** remain profitable in fold 1, so it is not a universal solution.

## Validation-selected advanced policy

| Fold | Pair | Validation-selected policy | OOS PnL | Sharpe | Pair screen |
|---:|---|---|---:|---:|---|
| 1 | CVX-XOM | Regime mixture-of-experts | -0.19870 | -1.5851 | Pass |
| 2 | CVX-XOM | Regime mixture-of-experts | +0.12337 | +1.1642 | Pass |
| 3 | KO-PEP | Regime mixture-of-experts | -0.01447 | -0.1704 | Fail / provisional |

Under the strict statistical deployment gate, fold 3 is flat because the pair screen fails.

## Strict-gate fixed-policy comparison

| Policy | Aggregate gated PnL | Positive valid folds | Negative valid folds | No-trade folds | Observed gated robustness |
|---|---:|---:|---:|---:|---|
| Random forest | **+0.14851** | 2 | 0 | 1 | **Yes** |
| Uncertainty consensus | **+0.12794** | 2 | 0 | 1 | **Yes** |
| Deep MLP | **+0.11103** | 2 | 0 | 1 | **Yes** |
| LSTM | +0.07057 | 1 | 1 | 1 | No |
| Regime mixture-of-experts | -0.07534 | 1 | 1 | 1 | No |
| GRU | -0.07948 | 1 | 1 | 1 | No |
| TCN | -0.14997 | 1 | 1 | 1 | No |

**Result:** more complex sequence/regime models improve some windows, especially fold 2, but do not produce a validation-selected universal solution. Fixed random forest, uncertainty consensus, and fixed deep MLP meet the observed gated-robustness criterion. The simpler candidate deployment architecture remains a strict statistical gate combined with a **fixed random forest or fixed deep MLP**.

## Universal-solution test

A strict universal solution would need to be chosen without test outcomes, produce positive OOS PnL and Sharpe in every fold, and never underperform the classical baseline. No tested ML/DL/advanced policy satisfies that requirement.

The more modest **observed gated-robustness** criterion is met by fixed random forest, uncertainty consensus, and fixed deep MLP: all three are profitable in every observed fold where the statistical screen passes and stay flat in the rejected fold. This remains a small-sample historical result, not a future-profit guarantee.

## ETF replication

The same LSTM, TCN, regime-aware MoE and uncertainty-consensus framework supports the pre-specified ETF universe. The completed verified public-data experiment selects IVV-VOO, which passes all configured screens in the available walk-forward fold. Because the supervised training labels contain no positive class, the dummy-majority fallback rejects all signals and remains flat with PnL 0.0 and Sharpe 0.0. This supports abstention but does not establish profitable or universal external replication.
