from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
import numpy as np
import pandas as pd

from .backtest import run_backtest, summarize_backtest
from .features import MODEL_FEATURE_COLUMNS, build_feature_frame
from .labels import build_convergence_labels
from .models import _classification_metrics, _choose_threshold, train_models
from .pairs import screen_pairs, select_top_pair
from .splits import TimeSplit
from .walkforward import make_walk_forward_splits


@dataclass
class GRUResult:
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    validation_probability: np.ndarray
    test_probability: np.ndarray
    validation_index: pd.Index
    test_index: pd.Index
    epochs_trained: int
    sequence_length: int


def _deep_cfg(cfg: dict) -> dict:
    base = {
        "sequence_length": 20,
        "hidden_size": 16,
        "num_layers": 1,
        "dropout": 0.0,
        "epochs": 80,
        "patience": 12,
        "batch_size": 16,
        "learning_rate": 0.003,
        "weight_decay": 0.001,
        "min_samples_per_split": 10,
    }
    base.update(cfg.get("deep_learning", {}))
    return base


def _signal_sequences(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Create trailing daily feature sequences ending at each labeled entry signal.

    Every feature in ``MODEL_FEATURE_COLUMNS`` is already lagged by the main pipeline, so the
    final element of each sequence is information available when the entry decision is made.
    """
    feature_values = features[MODEL_FEATURE_COLUMNS].astype(float)
    positions = {idx: i for i, idx in enumerate(feature_values.index)}
    xs: list[np.ndarray] = []
    ys: list[int] = []
    dates: list[pd.Timestamp] = []
    for idx, row in labels.sort_index().iterrows():
        pos = positions.get(idx)
        if pos is None or pos + 1 < sequence_length:
            continue
        seq = feature_values.iloc[pos - sequence_length + 1 : pos + 1].to_numpy(dtype=np.float32)
        if not np.isfinite(seq).all():
            continue
        xs.append(seq)
        ys.append(int(row["accept_signal"]))
        dates.append(pd.Timestamp(idx))
    if not xs:
        return (
            np.empty((0, sequence_length, len(MODEL_FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            pd.DatetimeIndex([]),
        )
    return np.stack(xs), np.asarray(ys, dtype=np.int64), pd.DatetimeIndex(dates)


def _split_sequence_samples(
    X: np.ndarray,
    y: np.ndarray,
    dates: pd.DatetimeIndex,
    split: TimeSplit,
) -> dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]]:
    split_sets = {
        "train": set(pd.DatetimeIndex(split.train_index)),
        "validation": set(pd.DatetimeIndex(split.validation_index)),
        "test": set(pd.DatetimeIndex(split.test_index)),
    }
    out: dict[str, tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]] = {}
    for name, allowed in split_sets.items():
        mask = np.array([date in allowed for date in dates], dtype=bool)
        out[name] = (X[mask], y[mask], dates[mask])
    return out


def _standardize_sequences(train_X: np.ndarray, *others: np.ndarray) -> tuple[np.ndarray, ...]:
    mean = train_X.mean(axis=(0, 1), keepdims=True)
    std = train_X.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return tuple(((arr - mean) / std).astype(np.float32) for arr in (train_X, *others))


def train_gru_classifier(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split: TimeSplit,
    cfg: dict,
) -> GRUResult:
    """Train a small recurrent neural network without touching the test set during tuning.

    The GRU is intentionally compact because labels exist only at valid entry signals. Training
    uses the formation set, early stopping and threshold selection use validation only, and the
    test period is evaluated once after model selection.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover - exercised when optional dependency is absent
        raise RuntimeError(
            "PyTorch is required for the GRU experiment. Install the final requirements.txt."
        ) from exc

    dcfg = _deep_cfg(cfg)
    sequence_length = int(dcfg["sequence_length"])
    X, y, dates = _signal_sequences(features, labels, sequence_length)
    parts = _split_sequence_samples(X, y, dates, split)
    train_X, train_y, train_idx = parts["train"]
    val_X, val_y, val_idx = parts["validation"]
    test_X, test_y, test_idx = parts["test"]
    minimum = int(dcfg.get("min_samples_per_split", 10))
    if min(len(train_X), len(val_X), len(test_X)) < minimum:
        raise ValueError(
            "Insufficient sequential entry signals for GRU: "
            f"train={len(train_X)}, validation={len(val_X)}, test={len(test_X)}"
        )
    if len(np.unique(train_y)) < 2:
        raise ValueError("GRU training requires both label classes in the training period.")

    train_X, val_X, test_X = _standardize_sequences(train_X, val_X, test_X)
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    class TinyGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(dcfg["hidden_size"])
            layers = int(dcfg["num_layers"])
            dropout = float(dcfg["dropout"]) if layers > 1 else 0.0
            self.gru = nn.GRU(
                input_size=len(MODEL_FEATURE_COLUMNS),
                hidden_size=hidden,
                num_layers=layers,
                batch_first=True,
                dropout=dropout,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, max(8, hidden // 2)),
                nn.ReLU(),
                nn.Dropout(float(dcfg.get("dropout", 0.0))),
                nn.Linear(max(8, hidden // 2), 1),
            )

        def forward(self, x: Any) -> Any:
            out, _ = self.gru(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    device = torch.device("cpu")
    model = TinyGRU().to(device)
    positives = max(1, int(train_y.sum()))
    negatives = max(1, int(len(train_y) - train_y.sum()))
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(dcfg["learning_rate"]),
        weight_decay=float(dcfg["weight_decay"]),
    )

    train_ds = TensorDataset(
        torch.from_numpy(train_X), torch.from_numpy(train_y.astype(np.float32))
    )
    loader_gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_ds,
        batch_size=min(int(dcfg["batch_size"]), len(train_ds)),
        shuffle=True,
        generator=loader_gen,
    )
    val_tensor = torch.from_numpy(val_X).to(device)
    val_target = torch.from_numpy(val_y.astype(np.float32)).to(device)

    best_state: dict[str, Any] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = int(dcfg["patience"])
    max_epochs = int(dcfg["epochs"])
    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_x), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_tensor), val_target).item())
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_left = int(dcfg["patience"])
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_prob = torch.sigmoid(model(torch.from_numpy(val_X).to(device))).cpu().numpy()
        test_prob = torch.sigmoid(model(torch.from_numpy(test_X).to(device))).cpu().numpy()

    thresholds = [float(v) for v in cfg["models"].get("threshold_grid", [0.5])]
    metric_name = str(cfg["models"].get("selection_metric", "f1"))
    val_series = pd.Series(val_y, index=val_idx)
    test_series = pd.Series(test_y, index=test_idx)
    threshold, val_metrics = _choose_threshold(val_series, val_prob, thresholds, metric_name)
    test_metrics = _classification_metrics(test_series, test_prob, threshold)
    return GRUResult(
        threshold=threshold,
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        validation_probability=val_prob,
        test_probability=test_prob,
        validation_index=val_idx,
        test_index=test_idx,
        epochs_trained=best_epoch,
        sequence_length=sequence_length,
    )


def _model_backtest_row(
    fold: int,
    pair: str,
    pair_passes: bool,
    pair_status: str,
    model_name: str,
    validation_f1: float,
    test_auc: float,
    test_f1: float,
    accept: pd.Series,
    test_features: pd.DataFrame,
    cfg: dict,
    baseline: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    results, trades = run_backtest(test_features, cfg, f"DL-WF{fold} {model_name}", accept)
    summary = summarize_backtest(results, trades, model_name, cfg, seed=seed, compute_ci=False)
    return {
        "fold": fold,
        "pair": pair,
        "pair_status": pair_status,
        "pair_passes_all": bool(pair_passes),
        "model": model_name,
        "validation_f1": float(validation_f1),
        "test_auc": float(test_auc),
        "test_f1": float(test_f1),
        "pnl": float(summary["cumulative_net_pnl"]),
        "sharpe": float(summary["sharpe"]),
        "max_drawdown": float(summary["max_drawdown"]),
        "trades": int(summary["closed_trades"]),
        "win_rate": float(summary["trade_win_rate"]),
        "baseline_pnl": float(baseline["cumulative_net_pnl"]),
        "baseline_sharpe": float(baseline["sharpe"]),
        "pnl_minus_baseline": float(summary["cumulative_net_pnl"] - baseline["cumulative_net_pnl"]),
        "profitable": bool(summary["cumulative_net_pnl"] > 0),
        "passed_and_profitable": bool(pair_passes and summary["cumulative_net_pnl"] > 0),
    }


def run_deep_walk_forward(prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compare classical ML, a deep MLP, and a GRU under identical walk-forward controls."""
    from copy import deepcopy

    all_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    splits = make_walk_forward_splits(prices.index, cfg)
    seed = int(cfg.get("project", {}).get("random_seed", 42))

    for fold_id, split in enumerate(splits, start=1):
        local = deepcopy(cfg)
        local["models"]["bootstrap_samples"] = 0
        base_algorithms = list(cfg.get("m7", {}).get("walk_forward", {}).get("algorithms", cfg["models"].get("algorithms", [])))
        tabular_algorithms = list(dict.fromkeys(base_algorithms + ["deep_mlp"]))
        local["models"]["algorithms"] = tabular_algorithms

        training_prices = prices.loc[prices.index.intersection(split.train_index)]
        candidates, _ = screen_pairs(training_prices, local)
        asset_a, asset_b, alpha, beta, pair_status = select_top_pair(candidates)
        selected = candidates.iloc[0]
        pair = f"{asset_a}-{asset_b}"
        pair_passes = bool(selected["passes_all"])
        available = prices.loc[: split.test_index[-1]]
        features = build_feature_frame(available, asset_a, asset_b, local, alpha, beta)
        labels = build_convergence_labels(features, local)
        test_features = features.loc[features.index.intersection(split.test_index)]
        baseline_results, baseline_trades = run_backtest(test_features, local, f"DL-WF{fold_id} baseline")
        baseline = summarize_backtest(
            baseline_results, baseline_trades, "baseline", local, seed=seed + fold_id, compute_ci=False
        )

        bundles, metrics, predictions, _, _ = train_models(features, labels, split, local)
        for offset, name in enumerate(bundles):
            metric = metrics.loc[metrics["model"] == name].iloc[0]
            row = _model_backtest_row(
                fold_id,
                pair,
                pair_passes,
                pair_status,
                name,
                float(metric["validation_f1"]),
                float(metric["test_auc"]),
                float(metric["test_f1"]),
                predictions[f"{name}_accept"],
                test_features,
                local,
                baseline,
                seed + 1000 + fold_id * 10 + offset,
            )
            all_rows.append(row)

        try:
            gru = train_gru_classifier(features, labels, split, local)
            gru_accept = pd.Series(
                (gru.test_probability >= gru.threshold).astype(int),
                index=gru.test_index,
                name="gru_accept",
            )
            row = _model_backtest_row(
                fold_id,
                pair,
                pair_passes,
                pair_status,
                "gru",
                float(gru.validation_metrics["f1"]),
                float(gru.test_metrics["auc"]),
                float(gru.test_metrics["f1"]),
                gru_accept,
                test_features,
                local,
                baseline,
                seed + 2000 + fold_id,
            )
            row["gru_epochs"] = gru.epochs_trained
            row["gru_sequence_length"] = gru.sequence_length
            all_rows.append(row)
        except Exception as exc:
            all_rows.append(
                {
                    "fold": fold_id,
                    "pair": pair,
                    "pair_status": pair_status,
                    "pair_passes_all": pair_passes,
                    "model": "gru",
                    "error": f"{type(exc).__name__}: {exc}",
                    "profitable": False,
                    "passed_and_profitable": False,
                }
            )

        fold_frame = pd.DataFrame([r for r in all_rows if r.get("fold") == fold_id and "validation_f1" in r])
        if not fold_frame.empty:
            # Validation F1 is the only selection criterion; test PnL is never used to choose the model.
            fold_frame = fold_frame.sort_values(["validation_f1", "model"], ascending=[False, True])
            winner = fold_frame.iloc[0].to_dict()
            winner["selected_by_validation"] = True
            selected_rows.append(winner)

    all_df = pd.DataFrame(all_rows)
    selected_df = pd.DataFrame(selected_rows)
    valid_selected = selected_df.dropna(subset=["pnl"]) if not selected_df.empty else selected_df
    universal = bool(
        len(valid_selected) == len(splits)
        and len(valid_selected) > 0
        and (valid_selected["pnl"] > 0).all()
        and (valid_selected["sharpe"] > 0).all()
        and (valid_selected["pnl_minus_baseline"] >= -1e-12).all()
    )
    universality = {
        "criterion": (
            "A universal ML/DL solution must be selected using validation information only, "
            "produce positive out-of-sample PnL and Sharpe in every walk-forward fold, and "
            "never underperform the classical baseline in cumulative PnL."
        ),
        "folds_evaluated": len(splits),
        "universal_solution_found": universal,
        "positive_pnl_folds": int((valid_selected["pnl"] > 0).sum()) if len(valid_selected) else 0,
        "positive_sharpe_folds": int((valid_selected["sharpe"] > 0).sum()) if len(valid_selected) else 0,
        "noninferior_to_baseline_folds": int((valid_selected["pnl_minus_baseline"] >= -1e-12).sum()) if len(valid_selected) else 0,
        "selected_models": selected_df[["fold", "pair", "model", "validation_f1", "pnl", "sharpe"]].to_dict("records") if not selected_df.empty else [],
    }
    return all_df, selected_df, universality


def run_deep_fixed_split(prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deep-learning comparison on the project's original fixed train/validation/test split."""
    from copy import deepcopy
    from .splits import make_time_split

    local = deepcopy(cfg)
    local["models"]["bootstrap_samples"] = 0
    local["models"]["algorithms"] = list(dict.fromkeys(list(cfg["models"].get("algorithms", [])) + ["deep_mlp"]))
    split = make_time_split(prices.index, local)
    training = prices.loc[prices.index.intersection(split.train_index)]
    candidates, _ = screen_pairs(training, local)
    a, b, alpha, beta, status = select_top_pair(candidates)
    selected_pair = candidates.iloc[0]
    features = build_feature_frame(prices, a, b, local, alpha, beta)
    labels = build_convergence_labels(features, local)
    test_features = features.loc[features.index.intersection(split.test_index)]
    baseline_results, baseline_trades = run_backtest(test_features, local, "baseline")
    baseline = summarize_backtest(baseline_results, baseline_trades, "baseline", local, compute_ci=False)
    bundles, metrics, predictions, _, _ = train_models(features, labels, split, local)
    rows: list[dict[str, Any]] = []
    for name in bundles:
        metric = metrics.loc[metrics["model"] == name].iloc[0]
        rows.append(
            _model_backtest_row(
                0,
                f"{a}-{b}",
                bool(selected_pair["passes_all"]),
                status,
                name,
                float(metric["validation_f1"]),
                float(metric["test_auc"]),
                float(metric["test_f1"]),
                predictions[f"{name}_accept"],
                test_features,
                local,
                baseline,
                9000,
            )
        )
    try:
        gru = train_gru_classifier(features, labels, split, local)
        accept = pd.Series((gru.test_probability >= gru.threshold).astype(int), index=gru.test_index)
        row = _model_backtest_row(
            0,
            f"{a}-{b}",
            bool(selected_pair["passes_all"]),
            status,
            "gru",
            float(gru.validation_metrics["f1"]),
            float(gru.test_metrics["auc"]),
            float(gru.test_metrics["f1"]),
            accept,
            test_features,
            local,
            baseline,
            9001,
        )
        row["gru_epochs"] = gru.epochs_trained
        rows.append(row)
    except Exception as exc:
        rows.append({"fold": 0, "pair": f"{a}-{b}", "model": "gru", "error": f"{type(exc).__name__}: {exc}"})
    result = pd.DataFrame(rows)
    selected = result.dropna(subset=["validation_f1"]).sort_values(["validation_f1", "model"], ascending=[False, True]).iloc[0].to_dict()
    summary = {
        "pair": f"{a}-{b}",
        "pair_status": status,
        "pair_passes_all": bool(selected_pair["passes_all"]),
        "baseline_pnl": baseline["cumulative_net_pnl"],
        "baseline_sharpe": baseline["sharpe"],
        "validation_selected_model": selected.get("model"),
        "selected_model_pnl": selected.get("pnl"),
        "selected_model_sharpe": selected.get("sharpe"),
    }
    return result, summary


def write_deep_outputs(
    output_dir: str | Path,
    fixed: pd.DataFrame | None = None,
    fixed_summary: dict[str, Any] | None = None,
    walkforward: pd.DataFrame | None = None,
    selected: pd.DataFrame | None = None,
    universality: dict[str, Any] | None = None,
) -> None:
    base = Path(output_dir)
    tables = base / "tables"
    figures = base / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if fixed is not None:
        fixed.to_csv(tables / "deep_fixed_split_model_comparison.csv", index=False)
    if fixed_summary is not None:
        (tables / "deep_fixed_split_summary.json").write_text(json.dumps(fixed_summary, indent=2, default=str), encoding="utf-8")
    if walkforward is not None:
        walkforward.to_csv(tables / "deep_walk_forward_model_comparison.csv", index=False)
        passed = walkforward.loc[walkforward.get("passed_and_profitable", False) == True].copy() if not walkforward.empty else pd.DataFrame()
        passed.to_csv(tables / "deep_passed_and_profitable.csv", index=False)

        # A strict deployment gate is a separate, pre-declared risk policy: if no pair passes
        # every configured statistical screen in a fold, the strategy stays flat instead of
        # trading a provisional best-available pair. This is useful for evaluating whether a
        # fixed model can avoid losses without weakening the pair screen.
        gate_rows: list[dict[str, Any]] = []
        for _, row in walkforward.dropna(subset=["model"]).iterrows():
            passes = bool(row.get("pair_passes_all", False))
            gated_pnl = float(row.get("pnl", 0.0)) if passes else 0.0
            gate_rows.append({
                "fold": int(row["fold"]),
                "pair": row["pair"],
                "model": row["model"],
                "pair_passes_all": passes,
                "original_pnl": row.get("pnl"),
                "strict_gate_pnl": gated_pnl,
                "strict_gate_trades": passes,
            })
        gate = pd.DataFrame(gate_rows)
        gate.to_csv(tables / "strict_statistical_gate_model_policy.csv", index=False)
        gate_summary_rows: list[dict[str, Any]] = []
        if not gate.empty:
            for model_name, group in gate.groupby("model"):
                traded = group.loc[group["strict_gate_trades"].astype(bool)]
                gate_summary_rows.append({
                    "model": model_name,
                    "aggregate_strict_gate_pnl": float(group["strict_gate_pnl"].sum()),
                    "negative_folds": int((group["strict_gate_pnl"] < 0).sum()),
                    "positive_folds": int((group["strict_gate_pnl"] > 0).sum()),
                    "no_trade_folds": int((~group["strict_gate_trades"].astype(bool)).sum()),
                    "profitable_valid_pair_folds": int((traded["strict_gate_pnl"] > 0).sum()),
                    "valid_pair_folds": int(len(traded)),
                    "nonnegative_in_all_observed_folds": bool((group["strict_gate_pnl"] >= -1e-12).all()),
                })
        pd.DataFrame(gate_summary_rows).sort_values(
            "aggregate_strict_gate_pnl", ascending=False
        ).to_csv(tables / "strict_statistical_gate_model_summary.csv", index=False)
    if selected is not None:
        selected.to_csv(tables / "deep_validation_selected_walk_forward.csv", index=False)
    if universality is not None:
        (tables / "deep_universality_summary.json").write_text(json.dumps(universality, indent=2, default=str), encoding="utf-8")
