# Deep Learning Extension - Final M7 Evidence

## Scope

The deep-learning extension preserves the original leakage controls and adds two neural models without replacing the classical ML benchmarks:

- `deep_mlp`: a compact three-hidden-layer feed-forward neural network (32-16-8 units) implemented with scikit-learn.
- `gru`: a small PyTorch gated recurrent unit using the trailing 20-day sequence of the nine lagged relationship-quality features.

The GRU is trained on the formation set, uses validation only for early stopping and threshold selection, and is evaluated once on the untouched test set. The original logistic regression, random forest, and gradient boosting models remain unchanged and are evaluated alongside the DL models.

## Five-pair real-data walk-forward results

The only pair that passes every configured statistical screen in the economically pre-specified five-pair hypothesis family is CVX-XOM, in folds 1 and 2.

- Fold 1: CVX-XOM passes all screens. Random forest produces PnL 0.047781 / Sharpe 0.734049 and the fixed deep MLP produces PnL 0.010307 / Sharpe 0.136817. The validation-selected model across ML/DL is gradient boosting, which reduces the baseline loss from -0.180209 to -0.000835 but is still slightly negative.
- Fold 2: CVX-XOM again passes all screens. Every tested ML/DL filter produces positive PnL; gradient boosting has the largest ex-post PnL at 0.102858 / Sharpe 0.940556, while the GRU has the strongest validation F1 and is therefore selected prospectively. The selected GRU produces PnL 0.100724 / Sharpe 0.901391.
- Fold 3: no pair passes every screen. KO-PEP is provisional. The validation-selected GRU is negative (-0.063813), so this fold rejects a universal-profitability claim.

The machine-readable table is `outputs_deep_m7_public/tables/deep_passed_and_profitable.csv`.

## Strict statistical gate

A stricter risk policy is also reported: do not trade a fold when no pair passes every statistical screen. Under this policy, no provisional pair is traded.

- Random forest: aggregate observed gated PnL 0.148505; positive in both valid-pair folds; no trade in fold 3; no negative observed fold.
- Deep MLP: aggregate observed gated PnL 0.111031; positive in both valid-pair folds; no trade in fold 3; no negative observed fold.

These are encouraging candidate policies, but they do **not** prove universal future profitability. The observed stability is based on only two statistically valid trading folds plus one no-trade fold, and the external ETF replication has not yet been numerically verified in this build environment.

## Universal ML/DL test

The pre-declared universality criterion for the validation-selected ML/DL system is: positive out-of-sample PnL and Sharpe in every walk-forward fold, and cumulative PnL not below the classical baseline in any fold, with model selection based on validation information only.

Result: **no universal ML/DL solution is demonstrated**.

Validation-selected results are:

1. Fold 1: gradient boosting, PnL -0.000835, Sharpe -0.010442.
2. Fold 2: GRU, PnL 0.100724, Sharpe 0.901391.
3. Fold 3: GRU, PnL -0.063813, Sharpe -0.716321.

The correct final conclusion is therefore stronger than the earlier ML-only version but still qualified: the DL extension produces additional profitable passed-and-profitable cases and a fixed deep MLP is positive in both statistically valid folds, yet a validation-selected ML/DL system is not universally profitable across all observed regimes.
