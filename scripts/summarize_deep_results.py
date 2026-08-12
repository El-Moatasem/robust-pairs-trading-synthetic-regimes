from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize final ML/DL walk-forward evidence.')
    parser.add_argument('--outputs', default='outputs_deep_m7_public')
    args = parser.parse_args()
    tables = Path(args.outputs) / 'tables'
    comparison = pd.read_csv(tables / 'deep_walk_forward_model_comparison.csv')
    selected = pd.read_csv(tables / 'deep_validation_selected_walk_forward.csv')
    passed = pd.read_csv(tables / 'deep_passed_and_profitable.csv')
    gated = pd.read_csv(tables / 'strict_statistical_gate_model_summary.csv')
    universal = json.loads((tables / 'deep_universality_summary.json').read_text())

    print('\nValidation-selected ML/DL model by fold')
    print(selected[['fold','pair','model','validation_f1','pnl','sharpe','pnl_minus_baseline']].to_string(index=False))
    print('\nAll passed-and-profitable model/pair cases')
    if passed.empty:
        print('None')
    else:
        print(passed[['fold','pair','model','pnl','sharpe','validation_f1']].to_string(index=False))
    print('\nStrict statistical-gate fixed-model policies')
    print(gated.to_string(index=False))
    print('\nUniversal validation-selected ML/DL solution found:', universal['universal_solution_found'])
    print(universal['criterion'])


if __name__ == '__main__':
    main()
