from __future__ import annotations

from dataclasses import asdict
from typing import Iterator

import numpy as np
import pandas as pd

from .backtest import run_backtest, summarize_backtest
from .features import build_feature_frame
from .labels import build_convergence_labels
from .models import train_models
from .pairs import screen_pairs, select_top_pair
from .splits import TimeSplit


def make_walk_forward_splits(index: pd.Index, cfg: dict) -> list[TimeSplit]:
    """Create expanding-window train/validation/test splits with purge and embargo.

    The training window expands by ``step_days``.  Validation and test windows are fixed-size.
    Pair screening and hedge-ratio estimation are repeated inside every fold using only that
    fold's training data.
    """
    wcfg = cfg.get("m7", {}).get("walk_forward", {})
    initial_train = int(wcfg.get("initial_train_days", 756))
    validation_days = int(wcfg.get("validation_days", 252))
    test_days = int(wcfg.get("test_days", 252))
    step_days = int(wcfg.get("step_days", test_days))
    max_folds = int(wcfg.get("max_folds", 10))
    purge = int(cfg["split"].get("purge_horizon_days", 10))
    embargo = int(cfg["split"].get("embargo_days", 2))

    splits: list[TimeSplit] = []
    train_cut = initial_train
    while len(splits) < max_folds:
        val_cut = train_cut + validation_days
        test_cut = val_cut + test_days
        if test_cut > len(index):
            break
        train_stop = max(1, train_cut - purge)
        val_start = min(len(index), train_cut + embargo)
        val_stop = max(val_start + 1, val_cut - purge)
        test_start = min(len(index) - 1, val_cut + embargo)
        train_idx = index[:train_stop]
        val_idx = index[val_start:val_stop]
        test_idx = index[test_start:test_cut]
        if min(len(train_idx), len(val_idx), len(test_idx)) < 20:
            break
        splits.append(
            TimeSplit(
                train_index=train_idx,
                validation_index=val_idx,
                test_index=test_idx,
                pair_selection_end=str(train_idx[-1].date()),
                train_end=str(train_idx[-1].date()),
                validation_start=str(val_idx[0].date()),
                validation_end=str(val_idx[-1].date()),
                test_start=str(test_idx[0].date()),
                test_end=str(test_idx[-1].date()),
                purge_horizon_days=purge,
                embargo_days=embargo,
            )
        )
        train_cut += step_days
    return splits


def run_walk_forward(prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the complete pair-screen -> ML-select -> OOS-backtest process in each fold."""
    fold_rows: list[dict] = []
    pair_rows: list[pd.DataFrame] = []
    splits = make_walk_forward_splits(prices.index, cfg)
    seed = int(cfg["project"].get("random_seed", 42))

    for fold_id, split in enumerate(splits, start=1):
        local_cfg = {**cfg, "models": {**cfg["models"], "bootstrap_samples": 0}}
        wf_algorithms = cfg.get("m7", {}).get("walk_forward", {}).get("algorithms")
        if wf_algorithms:
            local_cfg["models"]["algorithms"] = list(wf_algorithms)
        training_prices = prices.loc[prices.index.intersection(split.train_index)]
        candidate_pairs, pair_summary = screen_pairs(training_prices, local_cfg)
        annotated = candidate_pairs.copy()
        annotated.insert(0, "fold", fold_id)
        pair_rows.append(annotated)
        asset_a, asset_b, alpha, beta, pair_status = select_top_pair(candidate_pairs)
        selected = candidate_pairs.iloc[0]

        available_prices = prices.loc[: split.test_index[-1]]
        features = build_feature_frame(available_prices, asset_a, asset_b, local_cfg, alpha, beta)
        labels = build_convergence_labels(features, local_cfg)
        try:
            bundles, metrics, predictions, selected_model, supervised = train_models(features, labels, split, local_cfg)
            test_features = features.loc[features.index.intersection(split.test_index)]
            baseline_results, baseline_trades = run_backtest(test_features, local_cfg, f"WF{fold_id} baseline")
            baseline_summary = summarize_backtest(
                baseline_results, baseline_trades, f"WF{fold_id} baseline", local_cfg, seed + fold_id, compute_ci=False
            )
            accept = predictions[f"{selected_model}_accept"]
            ml_results, ml_trades = run_backtest(
                test_features, local_cfg, f"WF{fold_id} {selected_model}", accept_filter=accept
            )
            ml_summary = summarize_backtest(
                ml_results, ml_trades, f"WF{fold_id} {selected_model}", local_cfg, seed + 100 + fold_id, compute_ci=False
            )
            selected_metric = metrics.loc[metrics["model"] == selected_model].iloc[0]
            fold_rows.append(
                {
                    "fold": fold_id,
                    "train_end": split.train_end,
                    "validation_start": split.validation_start,
                    "validation_end": split.validation_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                    "selected_pair": f"{asset_a}-{asset_b}",
                    "pair_status": pair_status,
                    "pairs_passing_all": int(pair_summary["pairs_passing_all"]),
                    "abs_return_corr": float(selected["abs_return_corr"]),
                    "eg_pvalue": float(selected["eg_pvalue"]),
                    "residual_adf_pvalue": float(selected["residual_adf_pvalue"]),
                    "eg_fdr_pvalue": float(selected["eg_fdr_pvalue"]),
                    "estimated_half_life_days": float(selected["estimated_half_life_days"]),
                    "selected_model": selected_model,
                    "test_auc": float(selected_metric["test_auc"]),
                    "test_f1": float(selected_metric["test_f1"]),
                    "baseline_pnl": baseline_summary["cumulative_net_pnl"],
                    "baseline_sharpe": baseline_summary["sharpe"],
                    "baseline_max_drawdown": baseline_summary["max_drawdown"],
                    "ml_pnl": ml_summary["cumulative_net_pnl"],
                    "ml_sharpe": ml_summary["sharpe"],
                    "ml_max_drawdown": ml_summary["max_drawdown"],
                    "ml_trades": ml_summary["closed_trades"],
                    "ml_win_rate": ml_summary["trade_win_rate"],
                    "ml_minus_baseline_pnl": ml_summary["cumulative_net_pnl"] - baseline_summary["cumulative_net_pnl"],
                    "ml_minus_baseline_sharpe": ml_summary["sharpe"] - baseline_summary["sharpe"],
                    "passed_and_profitable": bool(selected["passes_all"] and ml_summary["cumulative_net_pnl"] > 0),
                }
            )
        except Exception as exc:
            fold_rows.append(
                {
                    "fold": fold_id,
                    "train_end": split.train_end,
                    "validation_start": split.validation_start,
                    "validation_end": split.validation_end,
                    "test_start": split.test_start,
                    "test_end": split.test_end,
                    "selected_pair": f"{asset_a}-{asset_b}",
                    "pair_status": pair_status,
                    "pairs_passing_all": int(pair_summary["pairs_passing_all"]),
                    "abs_return_corr": float(selected["abs_return_corr"]),
                    "eg_pvalue": float(selected["eg_pvalue"]),
                    "residual_adf_pvalue": float(selected["residual_adf_pvalue"]),
                    "eg_fdr_pvalue": float(selected["eg_fdr_pvalue"]),
                    "estimated_half_life_days": float(selected["estimated_half_life_days"]),
                    "selected_model": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "passed_and_profitable": False,
                }
            )

    folds = pd.DataFrame(fold_rows)
    pairs = pd.concat(pair_rows, ignore_index=True) if pair_rows else pd.DataFrame()
    return folds, pairs


def run_prespecified_pair_walk_forward(
    prices: pd.DataFrame, cfg: dict, pair_screens: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Sensitivity analysis for economically pre-specified pairs one at a time.

    Raw diagnostics are taken from the already-computed walk-forward pair screens. When a pair
    is treated as one pre-specified hypothesis, its BH-adjusted p-value equals its raw
    Engle-Granger p-value. This is a sensitivity analysis only and must not be used to choose a
    pair retroactively from test performance.
    """
    from copy import deepcopy

    if pair_screens is None:
        _, pair_screens = run_walk_forward(prices, cfg)
    pcfg = cfg["pair_selection"]
    raw_pass = (
        (pair_screens["abs_return_corr"] >= float(pcfg["min_abs_corr"]))
        & (pair_screens["eg_pvalue"] <= float(pcfg["max_eg_pvalue"]))
        & (pair_screens["residual_adf_pvalue"] <= float(pcfg["max_residual_adf_pvalue"]))
        & pair_screens["asset_a_plausibly_i1"].astype(bool)
        & pair_screens["asset_b_plausibly_i1"].astype(bool)
    )
    candidates = pair_screens.loc[raw_pass].copy()
    splits = make_walk_forward_splits(prices.index, cfg)
    rows: list[dict] = []
    for _, selected in candidates.iterrows():
        fold_id = int(selected["fold"])
        if fold_id < 1 or fold_id > len(splits):
            continue
        split = splits[fold_id - 1]
        local = deepcopy(cfg)
        wf_algorithms = cfg.get("m7", {}).get("walk_forward", {}).get("algorithms")
        if wf_algorithms:
            local["models"]["algorithms"] = list(wf_algorithms)
        local["models"]["bootstrap_samples"] = 0
        asset_a, asset_b = str(selected["asset_a"]), str(selected["asset_b"])
        alpha, beta = float(selected["alpha"]), float(selected["hedge_ratio"])
        features = build_feature_frame(prices.loc[: split.test_index[-1]], asset_a, asset_b, local, alpha, beta)
        labels = build_convergence_labels(features, local)
        try:
            _, metrics, predictions, selected_model, _ = train_models(features, labels, split, local)
        except Exception as exc:
            rows.append({"fold": fold_id, "pair": f"{asset_a}-{asset_b}", "error": f"{type(exc).__name__}: {exc}"})
            continue
        test_features = features.loc[features.index.intersection(split.test_index)]
        baseline_results, baseline_trades = run_backtest(test_features, local, "baseline")
        baseline = summarize_backtest(baseline_results, baseline_trades, "baseline", local, compute_ci=False)
        accept = predictions[f"{selected_model}_accept"]
        ml_results, ml_trades = run_backtest(test_features, local, selected_model, accept)
        ml = summarize_backtest(ml_results, ml_trades, selected_model, local, compute_ci=False)
        metric = metrics.loc[metrics["model"] == selected_model].iloc[0]
        rows.append(
            {
                "fold": fold_id,
                "pair": f"{asset_a}-{asset_b}",
                "test_start": split.test_start,
                "test_end": split.test_end,
                "eg_pvalue": float(selected["eg_pvalue"]),
                "residual_adf_pvalue": float(selected["residual_adf_pvalue"]),
                "fdr_pvalue_single_hypothesis": float(selected["eg_pvalue"]),
                "passes_i1": True,
                "estimated_half_life_days": float(selected["estimated_half_life_days"]),
                "selected_model": selected_model,
                "test_auc": float(metric["test_auc"]),
                "baseline_pnl": baseline["cumulative_net_pnl"],
                "baseline_sharpe": baseline["sharpe"],
                "ml_pnl": ml["cumulative_net_pnl"],
                "ml_sharpe": ml["sharpe"],
                "ml_trades": ml["closed_trades"],
                "ml_win_rate": ml["trade_win_rate"],
                "profitable_test": bool(ml["cumulative_net_pnl"] > 0),
            }
        )
    return pd.DataFrame(rows)
