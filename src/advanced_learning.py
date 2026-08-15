from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from .backtest import run_backtest, summarize_backtest
from .deep_learning import (
    _deep_cfg,
    _signal_sequences,
    _split_sequence_samples,
    _standardize_sequences,
    train_gru_classifier,
)
from .features import MODEL_FEATURE_COLUMNS, build_feature_frame
from .labels import build_convergence_labels
from .models import _catalog, _classification_metrics, _choose_threshold, _positive_probability
from .pairs import screen_pairs, select_top_pair
from .splits import TimeSplit, make_time_split
from .walkforward import make_walk_forward_splits


@dataclass
class SequenceModelResult:
    model_name: str
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    validation_probability: np.ndarray
    test_probability: np.ndarray
    validation_index: pd.DatetimeIndex
    test_index: pd.DatetimeIndex
    epochs_trained: int
    sequence_length: int


@dataclass
class ProbabilityBundle:
    name: str
    threshold: float
    validation_probability: pd.Series
    test_probability: pd.Series
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]


def _advanced_cfg(cfg: dict) -> dict:
    base = {
        "lstm_hidden_size": 16,
        "tcn_channels": 16,
        "tcn_kernel_size": 3,
        "tcn_dropout": 0.10,
        "regime_quantile": 0.75,
        "min_regime_validation_samples": 5,
        "consensus_vote_grid": [3, 4, 5],
        "expert_models": ["random_forest", "deep_mlp", "gru", "lstm", "tcn"],
    }
    base.update(cfg.get("advanced_learning", {}))
    return base


def _train_sequence_classifier(
    model_name: str,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split: TimeSplit,
    cfg: dict,
) -> SequenceModelResult:
    """Train a compact LSTM or causal temporal CNN using validation-only tuning.

    The sequence consists exclusively of lagged relationship-quality features ending on the
    entry-decision date. Training uses the formation set; validation is used for early stopping
    and probability-threshold selection; the test fold is untouched until final evaluation.
    """
    if model_name not in {"lstm", "tcn"}:
        raise ValueError(f"Unsupported sequence model: {model_name}")
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required for LSTM/TCN experiments.") from exc

    dcfg = _deep_cfg(cfg)
    acfg = _advanced_cfg(cfg)
    sequence_length = int(dcfg["sequence_length"])
    X, y, dates = _signal_sequences(features, labels, sequence_length)
    parts = _split_sequence_samples(X, y, dates, split)
    train_X, train_y, _ = parts["train"]
    val_X, val_y, val_idx = parts["validation"]
    test_X, test_y, test_idx = parts["test"]
    minimum = int(dcfg.get("min_samples_per_split", 10))
    if min(len(train_X), len(val_X), len(test_X)) < minimum:
        raise ValueError(
            f"Insufficient sequential signals for {model_name}: "
            f"train={len(train_X)}, validation={len(val_X)}, test={len(test_X)}"
        )
    if len(np.unique(train_y)) < 2:
        raise ValueError(f"{model_name} training requires both label classes.")

    train_X, val_X, test_X = _standardize_sequences(train_X, val_X, test_X)
    seed = int(cfg.get("project", {}).get("random_seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)

    class TinyLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hidden = int(acfg.get("lstm_hidden_size", dcfg.get("hidden_size", 16)))
            layers = int(dcfg.get("num_layers", 1))
            dropout = float(dcfg.get("dropout", 0.0)) if layers > 1 else 0.0
            self.rnn = nn.LSTM(
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
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    class CausalConv1d(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1) -> None:
            super().__init__()
            self.left_pad = (kernel - 1) * dilation
            self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel, dilation=dilation)

        def forward(self, x: Any) -> Any:
            import torch.nn.functional as F
            return self.conv(F.pad(x, (self.left_pad, 0)))

    class TinyTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = int(acfg.get("tcn_channels", 16))
            kernel = int(acfg.get("tcn_kernel_size", 3))
            dropout = float(acfg.get("tcn_dropout", dcfg.get("dropout", 0.1)))
            self.net = nn.Sequential(
                CausalConv1d(len(MODEL_FEATURE_COLUMNS), channels, kernel, dilation=1),
                nn.ReLU(),
                nn.Dropout(dropout),
                CausalConv1d(channels, channels, kernel, dilation=2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(channels, max(8, channels // 2)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(max(8, channels // 2), 1),
            )

        def forward(self, x: Any) -> Any:
            x = x.transpose(1, 2)
            return self.head(self.net(x)).squeeze(-1)

    device = torch.device("cpu")
    model = (TinyLSTM() if model_name == "lstm" else TinyTCN()).to(device)
    positives = max(1, int(train_y.sum()))
    negatives = max(1, int(len(train_y) - train_y.sum()))
    pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(dcfg["learning_rate"]),
        weight_decay=float(dcfg["weight_decay"]),
    )
    train_ds = TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y.astype(np.float32)))
    loader = DataLoader(
        train_ds,
        batch_size=min(int(dcfg["batch_size"]), len(train_ds)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    val_tensor = torch.from_numpy(val_X).to(device)
    val_target = torch.from_numpy(val_y.astype(np.float32)).to(device)

    best_state: dict[str, Any] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = int(dcfg["patience"])
    for epoch in range(1, int(dcfg["epochs"]) + 1):
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
    return SequenceModelResult(
        model_name=model_name,
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


def _tabular_probability_bundle(
    model_name: str,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split: TimeSplit,
    cfg: dict,
) -> ProbabilityBundle:
    joined = features[MODEL_FEATURE_COLUMNS].join(labels[["accept_signal", "realized_net_return"]], how="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    train = joined.loc[joined.index.intersection(split.train_index)]
    validation = joined.loc[joined.index.intersection(split.validation_index)]
    test = joined.loc[joined.index.intersection(split.test_index)]
    if min(len(train), len(validation), len(test)) < 10:
        raise ValueError("Insufficient labeled signals for advanced tabular experiment.")
    catalog = _catalog(int(cfg["models"].get("random_state", 42)))
    if model_name not in catalog:
        raise ValueError(f"Unknown tabular expert {model_name}")
    from sklearn.base import clone

    X_train, y_train = train[MODEL_FEATURE_COLUMNS], train["accept_signal"].astype(int)
    X_val, y_val = validation[MODEL_FEATURE_COLUMNS], validation["accept_signal"].astype(int)
    X_test, y_test = test[MODEL_FEATURE_COLUMNS], test["accept_signal"].astype(int)
    validation_model = clone(catalog[model_name])
    validation_model.fit(X_train, y_train)
    val_probability = _positive_probability(validation_model, X_val)
    threshold, val_metrics = _choose_threshold(
        y_val,
        val_probability,
        [float(v) for v in cfg["models"].get("threshold_grid", [0.5])],
        str(cfg["models"].get("selection_metric", "f1")),
    )
    final_model = clone(catalog[model_name])
    final_model.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_probability = _positive_probability(final_model, X_test)
    test_metrics = _classification_metrics(y_test, test_probability, threshold)
    return ProbabilityBundle(
        name=model_name,
        threshold=threshold,
        validation_probability=pd.Series(val_probability, index=X_val.index),
        test_probability=pd.Series(test_probability, index=X_test.index),
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def _sequence_probability_bundle(
    model_name: str,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split: TimeSplit,
    cfg: dict,
) -> ProbabilityBundle:
    if model_name == "gru":
        result = train_gru_classifier(features, labels, split, cfg)
        name = "gru"
    else:
        result = _train_sequence_classifier(model_name, features, labels, split, cfg)
        name = model_name
    return ProbabilityBundle(
        name=name,
        threshold=float(result.threshold),
        validation_probability=pd.Series(result.validation_probability, index=result.validation_index),
        test_probability=pd.Series(result.test_probability, index=result.test_index),
        validation_metrics=result.validation_metrics,
        test_metrics=result.test_metrics,
    )


def _regime_thresholds(features: pd.DataFrame, labels: pd.DataFrame, split: TimeSplit, cfg: dict) -> dict[str, float]:
    acfg = _advanced_cfg(cfg)
    q = float(acfg.get("regime_quantile", 0.75))
    train_idx = labels.index.intersection(split.train_index).intersection(features.index)
    train = features.loc[train_idx]
    return {
        "stress_high": float(train["regime_stress_proxy"].quantile(q)),
        "corr_low": float(train["rolling_corr"].quantile(1.0 - q)),
        "half_life_high": float(train["half_life"].quantile(q)),
        "persistence_high": float(train["deviation_persistence"].quantile(q)),
    }


def _assign_regime(features: pd.DataFrame, index: pd.Index, thresholds: dict[str, float]) -> pd.Series:
    frame = features.reindex(index)
    stress = (
        (frame["regime_stress_proxy"] >= thresholds["stress_high"])
        | (frame["rolling_corr"] <= thresholds["corr_low"])
    )
    slow = (~stress) & (
        (frame["half_life"] >= thresholds["half_life_high"])
        | (frame["deviation_persistence"] >= thresholds["persistence_high"])
    )
    regime = pd.Series("stable", index=frame.index, dtype="object")
    regime.loc[slow] = "slow_mean_reversion"
    regime.loc[stress] = "stress_or_breakdown"
    return regime


def _score_binary(y: pd.Series, pred: pd.Series) -> dict[str, float]:
    common = y.index.intersection(pred.index)
    if len(common) == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n": 0.0}
    probability = pred.loc[common].astype(float).to_numpy()
    metrics = _classification_metrics(y.loc[common].astype(int), probability, 0.5)
    return {"f1": float(metrics["f1"]), "precision": float(metrics["precision"]), "recall": float(metrics["recall"]), "n": float(len(common))}


def _make_regime_moe(
    bundles: dict[str, ProbabilityBundle],
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split: TimeSplit,
    cfg: dict,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Select a fixed expert per training-defined regime using validation F1 only."""
    acfg = _advanced_cfg(cfg)
    min_samples = int(acfg.get("min_regime_validation_samples", 5))
    thresholds = _regime_thresholds(features, labels, split, cfg)
    y_val = labels.loc[labels.index.intersection(split.validation_index), "accept_signal"].astype(int)
    val_index = y_val.index
    test_index = labels.index.intersection(split.test_index)
    val_regime = _assign_regime(features, val_index, thresholds)
    test_regime = _assign_regime(features, test_index, thresholds)

    global_ranking = sorted(
        bundles,
        key=lambda name: (float(bundles[name].validation_metrics.get("f1", 0.0)), name),
        reverse=True,
    )
    global_best = global_ranking[0]
    experts: dict[str, str] = {}
    validation_detail: dict[str, Any] = {}
    for regime in ["stable", "slow_mean_reversion", "stress_or_breakdown"]:
        idx = val_regime.index[val_regime == regime]
        scored: list[tuple[float, float, str, int]] = []
        for name, bundle in bundles.items():
            common = idx.intersection(bundle.validation_probability.index)
            if len(common) < min_samples:
                continue
            pred = (bundle.validation_probability.loc[common] >= bundle.threshold).astype(int)
            metrics = _classification_metrics(y_val.loc[common], pred.to_numpy(dtype=float), 0.5)
            scored.append((float(metrics["f1"]), float(metrics["precision"]), name, len(common)))
        if scored:
            scored.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
            chosen = scored[0][2]
            validation_detail[regime] = {
                "chosen_expert": chosen,
                "validation_f1": scored[0][0],
                "validation_precision": scored[0][1],
                "n": scored[0][3],
            }
        else:
            chosen = global_best
            validation_detail[regime] = {
                "chosen_expert": chosen,
                "fallback": "global_validation_f1",
                "n": int(len(idx)),
            }
        experts[regime] = chosen

    def compose(regime_series: pd.Series, which: str) -> pd.Series:
        out = pd.Series(0, index=regime_series.index, dtype=int)
        for date, regime in regime_series.items():
            name = experts[str(regime)]
            bundle = bundles[name]
            prob_series = bundle.validation_probability if which == "validation" else bundle.test_probability
            if date in prob_series.index:
                out.loc[date] = int(float(prob_series.loc[date]) >= bundle.threshold)
        return out

    val_accept = compose(val_regime, "validation")
    test_accept = compose(test_regime, "test")
    info = {
        "training_defined_regime_thresholds": thresholds,
        "experts_by_regime": experts,
        "validation_detail": validation_detail,
        "global_fallback_expert": global_best,
    }
    return val_accept, test_accept, info


def _make_uncertainty_consensus(
    bundles: dict[str, ProbabilityBundle],
    labels: pd.DataFrame,
    split: TimeSplit,
    cfg: dict,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    """Abstain unless enough fixed experts agree that the signal should be accepted.

    The required vote count is chosen on validation F1, with precision and then the larger vote
    count used as tie-breakers. Test outcomes are never used to choose the consensus strength.
    """
    y_val = labels.loc[labels.index.intersection(split.validation_index), "accept_signal"].astype(int)
    val_index = y_val.index
    test_index = labels.index.intersection(split.test_index)
    names = list(bundles)

    def votes(index: pd.Index, which: str) -> pd.Series:
        total = pd.Series(0, index=index, dtype=int)
        available = pd.Series(0, index=index, dtype=int)
        for name in names:
            bundle = bundles[name]
            probs = bundle.validation_probability if which == "validation" else bundle.test_probability
            common = index.intersection(probs.index)
            total.loc[common] += (probs.loc[common] >= bundle.threshold).astype(int)
            available.loc[common] += 1
        # If an expert is unavailable at a date, it does not contribute an affirmative vote.
        return total

    val_votes = votes(val_index, "validation")
    test_votes = votes(test_index, "test")
    grid = [int(v) for v in _advanced_cfg(cfg).get("consensus_vote_grid", [3, 4, 5])]
    grid = [v for v in grid if 1 <= v <= len(names)] or [max(1, len(names) // 2 + 1)]
    candidates: list[tuple[float, float, int, dict[str, float]]] = []
    for k in grid:
        pred = (val_votes >= k).astype(int)
        metrics = _classification_metrics(y_val, pred.to_numpy(dtype=float), 0.5)
        candidates.append((float(metrics["f1"]), float(metrics["precision"]), k, metrics))
    candidates.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    best = candidates[0]
    k = best[2]
    return (
        (val_votes >= k).astype(int),
        (test_votes >= k).astype(int),
        {
            "selected_min_votes": int(k),
            "n_experts": len(names),
            "expert_names": names,
            "validation_f1": best[0],
            "validation_precision": best[1],
            "candidate_vote_counts": [int(x) for x in grid],
        },
    )


def _backtest_policy(
    fold: int,
    pair: str,
    pair_status: str,
    pair_passes: bool,
    policy: str,
    validation_f1: float,
    test_accept: pd.Series,
    test_features: pd.DataFrame,
    cfg: dict,
    baseline: dict[str, Any],
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results, trades = run_backtest(test_features, cfg, f"ADV-WF{fold} {policy}", test_accept)
    summary = summarize_backtest(results, trades, policy, cfg, seed=seed, compute_ci=False)
    row: dict[str, Any] = {
        "fold": fold,
        "pair": pair,
        "pair_status": pair_status,
        "pair_passes_all": bool(pair_passes),
        "policy": policy,
        "validation_f1": float(validation_f1),
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
        "strict_gate_pnl": float(summary["cumulative_net_pnl"] if pair_passes else 0.0),
        "strict_gate_sharpe": float(summary["sharpe"] if pair_passes else 0.0),
        "strict_gate_trade": bool(pair_passes),
    }
    if extra:
        row.update(extra)
    return row


def _individual_accept(bundle: ProbabilityBundle) -> tuple[pd.Series, pd.Series]:
    val = (bundle.validation_probability >= bundle.threshold).astype(int)
    test = (bundle.test_probability >= bundle.threshold).astype(int)
    return val, test


def _build_expert_bundles(features: pd.DataFrame, labels: pd.DataFrame, split: TimeSplit, cfg: dict) -> dict[str, ProbabilityBundle]:
    requested = list(_advanced_cfg(cfg).get("expert_models", []))
    bundles: dict[str, ProbabilityBundle] = {}
    for name in requested:
        try:
            if name in {"random_forest", "deep_mlp", "logistic_regression", "gradient_boosting"}:
                bundles[name] = _tabular_probability_bundle(name, features, labels, split, cfg)
            elif name in {"gru", "lstm", "tcn"}:
                bundles[name] = _sequence_probability_bundle(name, features, labels, split, cfg)
        except Exception:
            # The caller records missing experts through the returned bundle set; one sparse
            # architecture must not invalidate the entire experiment.
            continue
    if len(bundles) < 2:
        raise ValueError(f"Need at least two successful experts; got {list(bundles)}")
    return bundles


def run_advanced_walk_forward(
    prices: pd.DataFrame, cfg: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run LSTM, TCN, regime-aware MoE and uncertainty-consensus experiments."""
    rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    splits = make_walk_forward_splits(prices.index, cfg)
    seed = int(cfg.get("project", {}).get("random_seed", 42))

    for fold_id, split in enumerate(splits, start=1):
        local = deepcopy(cfg)
        local["models"]["bootstrap_samples"] = 0
        training_prices = prices.loc[prices.index.intersection(split.train_index)]
        candidates, _ = screen_pairs(training_prices, local)
        a, b, alpha, beta, status = select_top_pair(candidates)
        selected = candidates.iloc[0]
        pair_passes = bool(selected["passes_all"])
        pair = f"{a}-{b}"
        features = build_feature_frame(prices.loc[: split.test_index[-1]], a, b, local, alpha, beta)
        labels = build_convergence_labels(features, local)
        test_features = features.loc[features.index.intersection(split.test_index)]
        baseline_results, baseline_trades = run_backtest(test_features, local, f"ADV-WF{fold_id} baseline")
        baseline = summarize_backtest(
            baseline_results, baseline_trades, "baseline", local, seed=seed + fold_id, compute_ci=False
        )
        bundles = _build_expert_bundles(features, labels, split, local)

        # Individual fixed experts, including the new LSTM and TCN.
        for offset, (name, bundle) in enumerate(bundles.items()):
            _, test_accept = _individual_accept(bundle)
            rows.append(
                _backtest_policy(
                    fold_id, pair, status, pair_passes, name,
                    float(bundle.validation_metrics.get("f1", 0.0)),
                    test_accept, test_features, local, baseline,
                    seed + 1000 + 50 * fold_id + offset,
                    extra={
                        "test_auc": float(bundle.test_metrics.get("auc", np.nan)),
                        "test_f1": float(bundle.test_metrics.get("f1", np.nan)),
                        "selected_threshold": float(bundle.threshold),
                    },
                )
            )

        # Regime-aware mixture of experts.
        val_moe, test_moe, regime_info = _make_regime_moe(bundles, features, labels, split, local)
        y_val = labels.loc[labels.index.intersection(split.validation_index), "accept_signal"].astype(int)
        moe_metrics = _classification_metrics(y_val.loc[y_val.index.intersection(val_moe.index)], val_moe.reindex(y_val.index).fillna(0).to_numpy(dtype=float), 0.5)
        rows.append(
            _backtest_policy(
                fold_id, pair, status, pair_passes, "regime_mixture_of_experts",
                float(moe_metrics["f1"]), test_moe, test_features, local, baseline,
                seed + 3000 + fold_id,
                extra={"policy_detail": json.dumps(regime_info, sort_keys=True)},
            )
        )
        for regime, expert in regime_info["experts_by_regime"].items():
            detail = regime_info["validation_detail"].get(regime, {})
            regime_rows.append({
                "fold": fold_id,
                "pair": pair,
                "regime": regime,
                "chosen_expert": expert,
                "validation_f1": detail.get("validation_f1"),
                "validation_precision": detail.get("validation_precision"),
                "validation_n": detail.get("n"),
                "fallback": detail.get("fallback", ""),
            })

        # Uncertainty-aware consensus/abstention policy.
        val_consensus, test_consensus, consensus_info = _make_uncertainty_consensus(bundles, labels, split, local)
        cons_metrics = _classification_metrics(y_val.loc[y_val.index.intersection(val_consensus.index)], val_consensus.reindex(y_val.index).fillna(0).to_numpy(dtype=float), 0.5)
        rows.append(
            _backtest_policy(
                fold_id, pair, status, pair_passes, "uncertainty_consensus",
                float(cons_metrics["f1"]), test_consensus, test_features, local, baseline,
                seed + 4000 + fold_id,
                extra={"policy_detail": json.dumps(consensus_info, sort_keys=True)},
            )
        )

    results = pd.DataFrame(rows)
    regimes = pd.DataFrame(regime_rows)

    # Validation-selected policy per fold. This includes advanced policies and experts, and is
    # selected strictly from validation F1; test PnL is not a selection input.
    selected_rows: list[dict[str, Any]] = []
    for fold, group in results.groupby("fold"):
        winner = group.sort_values(["validation_f1", "policy"], ascending=[False, True]).iloc[0].to_dict()
        winner["selected_by_validation"] = True
        selected_rows.append(winner)
    selected = pd.DataFrame(selected_rows)

    # Two notions are reported separately: strict universality and observed gated robustness.
    strict_universal = bool(
        len(selected) == len(splits)
        and (selected["pnl"] > 0).all()
        and (selected["sharpe"] > 0).all()
        and (selected["pnl_minus_baseline"] >= -1e-12).all()
    )
    gated_summary_rows: list[dict[str, Any]] = []
    for policy, group in results.groupby("policy"):
        valid = group.loc[group["pair_passes_all"].astype(bool)]
        gated_summary_rows.append({
            "policy": policy,
            "aggregate_strict_gate_pnl": float(group["strict_gate_pnl"].sum()),
            "positive_valid_pair_folds": int((valid["pnl"] > 0).sum()),
            "negative_valid_pair_folds": int((valid["pnl"] < 0).sum()),
            "valid_pair_folds": int(len(valid)),
            "no_trade_folds": int((~group["pair_passes_all"].astype(bool)).sum()),
            "nonnegative_all_observed_folds": bool((group["strict_gate_pnl"] >= -1e-12).all()),
            "positive_in_every_valid_pair_fold": bool(len(valid) > 0 and (valid["pnl"] > 0).all()),
            "observed_gated_robustness": bool(
                len(valid) >= 2
                and (valid["pnl"] > 0).all()
                and (group["strict_gate_pnl"] >= -1e-12).all()
            ),
        })
    gated_summary = pd.DataFrame(gated_summary_rows).sort_values("aggregate_strict_gate_pnl", ascending=False)
    robust_policies = gated_summary.loc[gated_summary["observed_gated_robustness"], "policy"].tolist()
    universality = {
        "strict_universal_definition": (
            "A validation-selected solution must have positive OOS PnL and Sharpe in every fold, "
            "and never underperform the classical baseline. A no-trade fold is not counted as profit."
        ),
        "strict_universal_solution_found": strict_universal,
        "selected_policies": selected[["fold", "pair", "policy", "validation_f1", "pnl", "sharpe", "pair_passes_all"]].to_dict("records"),
        "observed_gated_robustness_definition": (
            "A fixed policy has positive PnL in every fold where the statistical pair gate passes, "
            "and stays flat in rejected folds. This is an observed robustness result, not a universal future guarantee."
        ),
        "observed_gated_robust_policies": robust_policies,
        "external_replication_required_for_universal_claim": True,
    }
    return results, selected, gated_summary, universality, regimes


def run_advanced_fixed_split(prices: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate LSTM and TCN on the original fixed split for synthetic/public comparability."""
    local = deepcopy(cfg)
    local["models"]["bootstrap_samples"] = 0
    split = make_time_split(prices.index, local)
    training = prices.loc[prices.index.intersection(split.train_index)]
    candidates, _ = screen_pairs(training, local)
    a, b, alpha, beta, status = select_top_pair(candidates)
    selected = candidates.iloc[0]
    features = build_feature_frame(prices, a, b, local, alpha, beta)
    labels = build_convergence_labels(features, local)
    test_features = features.loc[features.index.intersection(split.test_index)]
    baseline_results, baseline_trades = run_backtest(test_features, local, "advanced baseline")
    baseline = summarize_backtest(baseline_results, baseline_trades, "baseline", local, compute_ci=False)
    rows: list[dict[str, Any]] = []
    for offset, name in enumerate(["lstm", "tcn"]):
        try:
            bundle = _sequence_probability_bundle(name, features, labels, split, local)
            test_accept = (bundle.test_probability >= bundle.threshold).astype(int)
            rows.append(
                _backtest_policy(
                    0, f"{a}-{b}", status, bool(selected["passes_all"]), name,
                    float(bundle.validation_metrics.get("f1", 0.0)),
                    test_accept, test_features, local, baseline, 7000 + offset,
                    extra={
                        "test_auc": float(bundle.test_metrics.get("auc", np.nan)),
                        "test_f1": float(bundle.test_metrics.get("f1", np.nan)),
                        "selected_threshold": float(bundle.threshold),
                    },
                )
            )
        except Exception as exc:
            rows.append({"fold": 0, "pair": f"{a}-{b}", "pair_status": status, "policy": name, "error": f"{type(exc).__name__}: {exc}"})
    result = pd.DataFrame(rows)
    summary = {
        "pair": f"{a}-{b}",
        "pair_status": status,
        "pair_passes_all": bool(selected["passes_all"]),
        "baseline_pnl": float(baseline["cumulative_net_pnl"]),
        "baseline_sharpe": float(baseline["sharpe"]),
    }
    return result, summary


def write_advanced_outputs(
    output_dir: str | Path,
    fixed: pd.DataFrame | None = None,
    fixed_summary: dict[str, Any] | None = None,
    walkforward: pd.DataFrame | None = None,
    selected: pd.DataFrame | None = None,
    gated_summary: pd.DataFrame | None = None,
    universality: dict[str, Any] | None = None,
    regimes: pd.DataFrame | None = None,
) -> None:
    base = Path(output_dir)
    tables = base / "tables"
    figures = base / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    if fixed is not None:
        fixed.to_csv(tables / "advanced_fixed_split_model_comparison.csv", index=False)
    if fixed_summary is not None:
        (tables / "advanced_fixed_split_summary.json").write_text(json.dumps(fixed_summary, indent=2, default=str), encoding="utf-8")
    if walkforward is not None:
        walkforward.to_csv(tables / "advanced_walk_forward_policy_comparison.csv", index=False)
        passed = walkforward.loc[walkforward["passed_and_profitable"].astype(bool)].copy()
        passed.to_csv(tables / "advanced_passed_and_profitable.csv", index=False)
    if selected is not None:
        selected.to_csv(tables / "advanced_validation_selected_walk_forward.csv", index=False)
    if gated_summary is not None:
        gated_summary.to_csv(tables / "advanced_strict_gate_policy_summary.csv", index=False)
    if universality is not None:
        (tables / "advanced_universality_summary.json").write_text(json.dumps(universality, indent=2, default=str), encoding="utf-8")
    if regimes is not None:
        regimes.to_csv(tables / "regime_mixture_expert_selection.csv", index=False)

    if walkforward is not None and not walkforward.empty:
        import matplotlib.pyplot as plt
        piv = walkforward.pivot_table(index="fold", columns="policy", values="pnl", aggfunc="first")
        ax = piv.plot(kind="bar", figsize=(10, 5))
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title("Advanced ML/DL walk-forward out-of-sample PnL")
        ax.set_xlabel("Walk-forward fold")
        ax.set_ylabel("Net log-spread PnL")
        ax.legend(title="Policy", fontsize=7)
        plt.tight_layout()
        plt.savefig(figures / "advanced_walk_forward_pnl.png", dpi=160)
        plt.close()
    if gated_summary is not None and not gated_summary.empty:
        frame = gated_summary.sort_values("aggregate_strict_gate_pnl", ascending=False)
        ax = frame.plot(x="policy", y="aggregate_strict_gate_pnl", kind="bar", legend=False, figsize=(9, 4.8))
        ax.axhline(0.0, linewidth=0.8)
        ax.set_title("Strict statistical-gate aggregate PnL by fixed policy")
        ax.set_xlabel("Policy")
        ax.set_ylabel("Aggregate gated PnL")
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(figures / "advanced_strict_gate_summary.png", dpi=160)
        plt.close()
