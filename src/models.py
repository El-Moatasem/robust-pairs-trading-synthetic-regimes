from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import MODEL_FEATURE_COLUMNS
from .splits import TimeSplit


@dataclass
class ModelBundle:
    name: str
    model: Any
    threshold: float
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    feature_importance: pd.DataFrame


def _catalog(random_state: int) -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=6,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=3,
            min_samples_leaf=8,
            random_state=random_state,
        ),
    }


def _positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        if probabilities.shape[1] == 1:
            only_class = int(getattr(model, "classes_", [0])[0])
            return np.ones(len(X)) if only_class == 1 else np.zeros(len(X))
        classes = list(model.classes_)
        return probabilities[:, classes.index(1)] if 1 in classes else np.zeros(len(X))
    scores = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-np.asarray(scores)))


def _classification_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    auc = float("nan") if y_true.nunique() < 2 else float(roc_auc_score(y_true, probability))
    return {
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "auc": auc,
        "brier": float(brier_score_loss(y_true, probability)),
        "class_prevalence": float(y_true.mean()),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
        "tp": float(tp),
        "n_observations": float(len(y_true)),
    }


def _choose_threshold(y_val: pd.Series, probability: np.ndarray, thresholds: list[float], metric: str) -> tuple[float, dict[str, float]]:
    best_threshold = thresholds[0]
    best_metrics = _classification_metrics(y_val, probability, best_threshold)
    best_score = best_metrics.get(metric, best_metrics["f1"])
    for threshold in thresholds[1:]:
        metrics = _classification_metrics(y_val, probability, threshold)
        score = metrics.get(metric, metrics["f1"])
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12 and metrics["precision"] > best_metrics["precision"]
        ):
            best_threshold = threshold
            best_metrics = metrics
            best_score = score
    return float(best_threshold), best_metrics


def _bootstrap_auc_ci(y: pd.Series, probability: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if y.nunique() < 2 or samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    y_values = y.to_numpy()
    for _ in range(samples):
        idx = rng.integers(0, len(y_values), len(y_values))
        sampled_y = y_values[idx]
        if len(np.unique(sampled_y)) < 2:
            continue
        values.append(float(roc_auc_score(sampled_y, probability[idx])))
    if len(values) < 10:
        return float("nan"), float("nan")
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _feature_importance(name: str, model: Any) -> pd.DataFrame:
    if name == "logistic_regression" and hasattr(model, "named_steps"):
        values = np.abs(model.named_steps["classifier"].coef_[0])
    elif hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    else:
        values = np.zeros(len(MODEL_FEATURE_COLUMNS))
    return pd.DataFrame({"feature": MODEL_FEATURE_COLUMNS, "importance": values}).sort_values(
        "importance", ascending=False
    )


def train_models(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    split: TimeSplit,
    cfg: dict,
) -> tuple[dict[str, ModelBundle], pd.DataFrame, pd.DataFrame, str, dict]:
    joined = features[MODEL_FEATURE_COLUMNS].join(labels[["accept_signal", "realized_net_return"]], how="inner")
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()

    train = joined.loc[joined.index.intersection(split.train_index)]
    validation = joined.loc[joined.index.intersection(split.validation_index)]
    test = joined.loc[joined.index.intersection(split.test_index)]
    if min(len(train), len(validation), len(test)) < 10:
        raise ValueError(
            "Insufficient entry signals after purging: "
            f"train={len(train)}, validation={len(validation)}, test={len(test)}. "
            "Increase the sample length or lower the entry threshold."
        )

    mcfg = cfg["models"]
    random_state = int(mcfg.get("random_state", 42))
    thresholds = [float(value) for value in mcfg.get("threshold_grid", [0.5])]
    selection_metric = str(mcfg.get("selection_metric", "f1"))
    catalog = _catalog(random_state)
    requested = [name for name in mcfg.get("algorithms", catalog.keys()) if name in catalog]
    if not requested:
        requested = ["logistic_regression"]

    X_train, y_train = train[MODEL_FEATURE_COLUMNS], train["accept_signal"].astype(int)
    X_val, y_val = validation[MODEL_FEATURE_COLUMNS], validation["accept_signal"].astype(int)
    X_test, y_test = test[MODEL_FEATURE_COLUMNS], test["accept_signal"].astype(int)

    if y_train.nunique() < 2:
        requested = ["dummy_majority"]
        catalog["dummy_majority"] = DummyClassifier(strategy="prior")

    bundles: dict[str, ModelBundle] = {}
    metric_rows: list[dict] = []
    prediction_frame = pd.DataFrame(index=X_test.index)
    prediction_frame["actual"] = y_test
    prediction_frame["realized_net_return"] = test["realized_net_return"]

    for offset, name in enumerate(requested):
        base_model = catalog[name]
        validation_model = clone(base_model)
        validation_model.fit(X_train, y_train)
        val_probability = _positive_probability(validation_model, X_val)
        threshold, val_metrics = _choose_threshold(y_val, val_probability, thresholds, selection_metric)

        final_model = clone(base_model)
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train, y_val])
        final_model.fit(X_train_val, y_train_val)
        test_probability = _positive_probability(final_model, X_test)
        test_metrics = _classification_metrics(y_test, test_probability, threshold)
        ci_low, ci_high = _bootstrap_auc_ci(
            y_test,
            test_probability,
            samples=int(mcfg.get("bootstrap_samples", 250)),
            seed=random_state + offset,
        )
        test_metrics["auc_ci_low"] = ci_low
        test_metrics["auc_ci_high"] = ci_high

        prediction_frame[f"{name}_probability"] = test_probability
        prediction_frame[f"{name}_accept"] = (test_probability >= threshold).astype(int)
        bundles[name] = ModelBundle(
            name=name,
            model=final_model,
            threshold=threshold,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
            feature_importance=_feature_importance(name, final_model),
        )
        metric_rows.append(
            {
                "model": name,
                "selected_threshold_from_validation": threshold,
                **{f"validation_{key}": value for key, value in val_metrics.items()},
                **{f"test_{key}": value for key, value in test_metrics.items()},
            }
        )

    metrics_table = pd.DataFrame(metric_rows)
    selection_column = f"validation_{selection_metric}"
    if selection_column not in metrics_table:
        selection_column = "validation_f1"
    selected_model = str(metrics_table.sort_values(selection_column, ascending=False).iloc[0]["model"])
    split_summary = {
        "labeled_signals_total": int(len(joined)),
        "train_signals": int(len(train)),
        "validation_signals": int(len(validation)),
        "test_signals": int(len(test)),
        "train_prevalence": float(y_train.mean()),
        "validation_prevalence": float(y_val.mean()),
        "test_prevalence": float(y_test.mean()),
        "model_selection_basis": f"highest {selection_metric} on validation only",
        "selected_model": selected_model,
    }
    return bundles, metrics_table, prediction_frame, selected_model, split_summary
