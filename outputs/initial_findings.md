# Initial Findings and Temporary Results

The current codebase was run on reproducible daily sample data so that the pipeline can be reviewed without internet access. These results are temporary and intended to demonstrate code development, not final investment performance.

- Best temporary strategy by Sharpe: Baseline z-score with Sharpe -0.289.

- Best temporary model by F1: random_forest with F1 0.627.

- Synthetic scenarios with the largest spread volatility are: scenario_010_stress, scenario_005_stress, scenario_015_stress.

- The initial results suggest the project can already select candidate pairs, construct spread features, produce accept/reject labels, train ML filters, run baseline/ML-filtered backtests, and generate robustness scenarios.
