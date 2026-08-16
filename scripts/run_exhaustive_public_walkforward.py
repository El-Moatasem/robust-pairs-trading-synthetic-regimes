from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.backtest import run_backtest, summarize_backtest
from src.data import load_prices
from src.features import build_feature_frame
from src.labels import build_convergence_labels
from src.models import train_models
from src.pairs import screen_pairs
from src.utils import load_config, save_json, set_seed
from src.walkforward import make_walk_forward_splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Exhaustively screen all unordered pairs in the public ticker universe in each '
            'walk-forward fold and backtest raw-diagnostic candidates out of sample. '
            'BH-FDR is computed across the full pair universe in each fold.'
        )
    )
    parser.add_argument('--config', default='config/config_m7_public.yaml')
    parser.add_argument('--output-dir', default='outputs_exhaustive_public')
    parser.add_argument('--require-public-data', action='store_true')
    parser.add_argument('--backtest-raw', action='store_true', help='Also backtest candidates that pass raw diagnostics but fail the full-universe FDR gate; this is slower and exploratory.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = deepcopy(cfg)
    cfg['pair_selection'].pop('candidate_pairs', None)
    cfg['pair_selection']['top_n'] = 100000
    cfg['models']['bootstrap_samples'] = 0
    cfg['backtest']['bootstrap_samples'] = 0
    seed = int(cfg['project'].get('random_seed', 42))
    set_seed(seed)

    prices, data_summary = load_prices(cfg, require_public=args.require_public_data)
    splits = make_walk_forward_splits(prices.index, cfg)
    out = Path(args.output_dir)
    tables = out / 'tables'
    tables.mkdir(parents=True, exist_ok=True)

    all_screens: list[pd.DataFrame] = []
    bt_rows: list[dict] = []
    pcfg = cfg['pair_selection']
    for fold_id, split in enumerate(splits, start=1):
        training = prices.loc[prices.index.intersection(split.train_index)]
        screen, summary = screen_pairs(training, cfg, return_all=True)
        screen.insert(0, 'fold', fold_id)
        all_screens.append(screen)

        # Exploratory only: candidates passing every raw diagnostic except the global BH-FDR gate.
        raw_pass = (
            screen['passes_corr'].astype(bool)
            & screen['passes_eg'].astype(bool)
            & screen['passes_residual_adf'].astype(bool)
            & screen['passes_i1'].astype(bool)
        )
        if not args.backtest_raw:
            continue
        for _, row in screen.loc[raw_pass].iterrows():
            asset_a, asset_b = str(row['asset_a']), str(row['asset_b'])
            alpha, beta = float(row['alpha']), float(row['hedge_ratio'])
            local = deepcopy(cfg)
            wf_algs = cfg.get('m7', {}).get('walk_forward', {}).get('algorithms')
            if wf_algs:
                local['models']['algorithms'] = list(wf_algs)
            available = prices.loc[: split.test_index[-1]]
            features = build_feature_frame(available, asset_a, asset_b, local, alpha, beta)
            labels = build_convergence_labels(features, local)
            try:
                _, metrics, predictions, selected_model, _ = train_models(features, labels, split, local)
                test_features = features.loc[features.index.intersection(split.test_index)]
                base_res, base_trades = run_backtest(test_features, local, 'baseline')
                base = summarize_backtest(base_res, base_trades, 'baseline', local, compute_ci=False)
                accept = predictions[f'{selected_model}_accept']
                ml_res, ml_trades = run_backtest(test_features, local, selected_model, accept)
                ml = summarize_backtest(ml_res, ml_trades, selected_model, local, compute_ci=False)
                metric = metrics.loc[metrics['model'] == selected_model].iloc[0]
                bt_rows.append({
                    'fold': fold_id,
                    'pair': f'{asset_a}-{asset_b}',
                    'test_start': split.test_start,
                    'test_end': split.test_end,
                    'eg_p': float(row['eg_pvalue']),
                    'fdr45': float(row['eg_fdr_pvalue']),
                    'residual_adf_p': float(row['residual_adf_pvalue']),
                    'half_life': float(row['estimated_half_life_days']),
                    'full_45pair_fdr_pass': bool(row['passes_all']),
                    'model': selected_model,
                    'auc': float(metric['test_auc']),
                    'f1': float(metric['test_f1']),
                    'base_pnl': base['cumulative_net_pnl'],
                    'base_sharpe': base['sharpe'],
                    'ml_pnl': ml['cumulative_net_pnl'],
                    'ml_sharpe': ml['sharpe'],
                    'ml_dd': ml['max_drawdown'],
                    'ml_trades': ml['closed_trades'],
                    'win': ml['trade_win_rate'],
                    'delta_pnl': ml['cumulative_net_pnl'] - base['cumulative_net_pnl'],
                })
            except Exception as exc:
                bt_rows.append({
                    'fold': fold_id,
                    'pair': f'{asset_a}-{asset_b}',
                    'eg_p': float(row['eg_pvalue']),
                    'fdr45': float(row['eg_fdr_pvalue']),
                    'residual_adf_p': float(row['residual_adf_pvalue']),
                    'half_life': float(row['estimated_half_life_days']),
                    'full_45pair_fdr_pass': bool(row['passes_all']),
                    'error': f'{type(exc).__name__}: {exc}',
                })

    screens = pd.concat(all_screens, ignore_index=True) if all_screens else pd.DataFrame()
    backtests = pd.DataFrame(bt_rows)
    screens.to_csv(tables / 'all45_realdata_walkforward_pair_screens.csv', index=False)
    backtests.to_csv(tables / 'all45_realdata_exploratory_raw_pass_backtests.csv', index=False)
    profitable_raw = backtests.loc[(backtests.get('ml_pnl', 0) > 0)].copy() if len(backtests) else pd.DataFrame()
    profitable_raw.to_csv(tables / 'raw_diagnostic_positive_oos_cases.csv', index=False)

    by_fold = []
    for fold_id, group in screens.groupby('fold') if len(screens) else []:
        by_fold.append({
            'fold': int(fold_id),
            'pairs_tested': int(len(group)),
            'pairs_passing_all_45pair_fdr': int(group['passes_all'].sum()),
            'minimum_raw_eg_p': float(group['eg_pvalue'].min()),
            'minimum_bh_fdr_p': float(group['eg_fdr_pvalue'].min()),
        })
    summary = {
        'data_source': data_summary.source,
        'synthetic': bool(data_summary.is_synthetic),
        'tickers': list(prices.columns),
        'pair_universe_size': int(len(prices.columns) * (len(prices.columns) - 1) / 2),
        'walk_forward_folds': int(len(splits)),
        'pair_fold_hypotheses': int(len(screens)),
        'full_fdr_pass_cases': int(screens['passes_all'].sum()) if len(screens) else 0,
        'raw_diagnostic_backtests': int(len(backtests)),
        'raw_diagnostic_positive_oos_cases': int((backtests.get('ml_pnl', pd.Series(dtype=float)) > 0).sum()) if len(backtests) else 0,
        'folds': by_fold,
        'interpretation': (
            'The raw-diagnostic backtests are exploratory. A positive OOS PnL does not constitute '
            'a statistically validated pair when the pair fails the BH-FDR gate across the full 45-pair universe.'
        ),
    }
    save_json(summary, out / 'exhaustive_summary.json')

    print('Exhaustive public walk-forward completed successfully.')
    print(f'Data source: {data_summary.source}')
    print(f'Synthetic data: {data_summary.is_synthetic}')
    print(f'Folds: {len(splits)}')
    print(f'Pair-fold screens: {len(screens)}')
    print(f'Full 45-pair FDR pass cases: {summary["full_fdr_pass_cases"]}')
    print(f'Raw-diagnostic candidates backtested: {len(backtests)}')
    if not args.backtest_raw:
        print('Raw-diagnostic backtests skipped; re-run with --backtest-raw for the exploratory economic pass.')
    print(f'Raw-diagnostic positive OOS cases: {summary["raw_diagnostic_positive_oos_cases"]}')
    print(f'Outputs: {out.resolve()}')


if __name__ == '__main__':
    main()
