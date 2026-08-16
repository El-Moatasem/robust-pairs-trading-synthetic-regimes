from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print('\n$', ' '.join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description='Run all final capstone experiments.')
    parser.add_argument('--with-etf', action='store_true', help='Also run strict real-data ETF replication.')
    args = parser.parse_args()
    py = sys.executable

    run(py, '-m', 'pytest', '-q')
    run(py, 'run_pipeline.py', '--config', 'config/config.yaml')
    run(py, 'scripts/verify_outputs.py', '--outputs', 'outputs')
    run(py, 'run_pipeline.py', '--config', 'config/config_public.yaml', '--require-public-data')
    run(py, 'scripts/verify_outputs.py', '--outputs', 'outputs_public')
    run(py, 'run_pipeline.py', '--config', 'config/config_m7_public.yaml', '--require-public-data')
    run(py, 'scripts/run_m7_analysis.py', '--config', 'config/config_m7_public.yaml', '--require-public-data')
    run(py, 'scripts/run_exhaustive_public_walkforward.py', '--config', 'config/config_m7_public.yaml', '--require-public-data')

    # Deep-learning extension: deep MLP + sequence-aware GRU. These experiments preserve
    # the original ML results and add a separate validation-only DL comparison.
    run(py, 'scripts/run_deep_learning_analysis.py', '--config', 'config/config.yaml', '--output-dir', 'outputs_deep_synthetic')
    run(py, 'scripts/run_deep_learning_analysis.py', '--config', 'config/config_public.yaml', '--require-public-data', '--output-dir', 'outputs_deep_public')
    run(py, 'scripts/run_deep_learning_analysis.py', '--config', 'config/config_m7_public.yaml', '--require-public-data', '--walk-forward', '--output-dir', 'outputs_deep_m7_public')
    run(py, 'scripts/summarize_deep_results.py', '--outputs', 'outputs_deep_m7_public')

    # Advanced robustness extension: LSTM, causal TCN, training-defined regime mixture-of-experts,
    # and an uncertainty-aware consensus/abstention policy.
    run(py, 'scripts/run_advanced_learning_analysis.py', '--config', 'config/config.yaml', '--output-dir', 'outputs_advanced_synthetic')
    run(py, 'scripts/run_advanced_learning_analysis.py', '--config', 'config/config_public.yaml', '--require-public-data', '--output-dir', 'outputs_advanced_public')
    run(py, 'scripts/run_advanced_learning_analysis.py', '--config', 'config/config_m7_public.yaml', '--require-public-data', '--walk-forward', '--output-dir', 'outputs_advanced_m7_public')
    run(py, 'scripts/summarize_advanced_results.py', '--outputs', 'outputs_advanced_m7_public')

    if args.with_etf:
        run(py, 'run_pipeline.py', '--config', 'config/config_m7_etf.yaml', '--require-public-data')
        run(py, 'scripts/verify_outputs.py', '--outputs', 'outputs_m7_etf')
        run(py, 'scripts/run_m7_analysis.py', '--config', 'config/config_m7_etf.yaml', '--require-public-data')
        run(py, 'scripts/run_deep_learning_analysis.py', '--config', 'config/config_m7_etf.yaml', '--require-public-data', '--walk-forward', '--output-dir', 'outputs_deep_m7_etf')
        run(py, 'scripts/run_advanced_learning_analysis.py', '--config', 'config/config_m7_etf.yaml', '--require-public-data', '--walk-forward', '--output-dir', 'outputs_advanced_m7_etf')
    else:
        print('\nETF replication skipped. Re-run with --with-etf in a network-enabled environment, or after placing data/raw/etf_prices.csv.')


if __name__ == '__main__':
    main()
