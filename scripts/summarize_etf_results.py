from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description='Summarize strict real-data ETF replication outputs.')
    parser.add_argument('--outputs', default='outputs_m7_etf')
    args = parser.parse_args()
    base = Path(args.outputs)
    wf = base / 'm7' / 'tables' / 'walk_forward_summary.csv'
    pairs = base / 'm7' / 'tables' / 'walk_forward_pair_screens.csv'
    if not wf.exists() or not pairs.exists():
        raise SystemExit(
            f'ETF results not found under {base}. Run run_pipeline.py and scripts/run_m7_analysis.py with config/config_m7_etf.yaml first.'
        )
    w = pd.read_csv(wf)
    p = pd.read_csv(pairs)
    print('=== ETF walk-forward selected-pair summary ===')
    cols = [c for c in ['fold','selected_pair','pair_status','pairs_passing_all','eg_pvalue','eg_fdr_pvalue','estimated_half_life_days','selected_model','ml_pnl','ml_sharpe','passed_and_profitable'] if c in w.columns]
    print(w[cols].to_string(index=False))
    print('\n=== ETF pairs passing all configured tests ===')
    passed = p[p['passes_all'].astype(bool)] if 'passes_all' in p else pd.DataFrame()
    if len(passed):
        cols2=[c for c in ['fold','asset_a','asset_b','abs_return_corr','eg_pvalue','residual_adf_pvalue','eg_fdr_pvalue','estimated_half_life_days'] if c in passed.columns]
        print(passed[cols2].to_string(index=False))
    else:
        print('No ETF pair passed all configured tests in the reported folds.')
    print('\n=== Passed and profitable selected folds ===')
    pp=w[w.get('passed_and_profitable', False).astype(bool)] if 'passed_and_profitable' in w else pd.DataFrame()
    if len(pp):
        print(pp[cols].to_string(index=False))
    else:
        print('No selected ETF fold both passed all configured tests and produced positive OOS ML PnL.')


if __name__ == '__main__':
    main()
