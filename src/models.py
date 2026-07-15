from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "zscore",
    "abs_zscore",
    "spread_change",
    "spread_volatility",
    "rolling_corr",
    "spread_drawdown",
    "half_life",
    "deviation_persistence",
    "regime_stress_proxy",
]


@dataclass
class ModelBundle:
    name: str
    model: object
    metrics: Dict[str, float]
    feature_importance: pd.DataFrame


def train_test_split_time(
    features: pd.DataFrame,
    labels: pd.Series,
    train_fraction: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    common = features.join(labels, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    if len(common) < 40:
        raise ValueError("Not enough labeled observations to train and test models. Increase sample size or horizon.")
    split = int(len(common) * train_fraction)
    split = min(max(20, split), len(common) - 10)
    X = common[FEATURE_COLUMNS]
    y = common[labels.name]
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def _metrics(y_true: pd.Series, proba: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    if len(np.unique(y_true)) < 2 or len(np.unique(proba)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, proba))
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": auc,
    }


def _model_catalog(random_state: int) -> Dict[str, object]:
    catalog: Dict[str, object] = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=random_state)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=5,
            min_samples_leaf=8,
            random_state=random_state,
            class_weight="balanced",
        ),
    }
    try:
        from xgboost import XGBClassifier  # type: ignore

        catalog["xgboost"] = XGBClassifier(
            n_estimators=150,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=random_state,
        )
    except Exception:
        catalog["xgboost_fallback_gradient_boosting"] = GradientBoostingClassifier(random_state=random_state)
    return catalog


def _feature_importance(name: str, model: object) -> pd.DataFrame:
    if name == "logistic_regression" and hasattr(model, "named_steps"):
        coefs = model.named_steps["clf"].coef_[0]
        values = np.abs(coefs)
    elif hasattr(model, "feature_importances_"):
        values = getattr(model, "feature_importances_")
    else:
        values = np.zeros(len(FEATURE_COLUMNS))
    return pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": values}).sort_values("importance", ascending=False)


def train_models(features: pd.DataFrame, labels: pd.Series, cfg: dict) -> tuple[Dict[str, ModelBundle], pd.DataFrame, pd.DataFrame]:
    mcfg = cfg["models"]
    train_fraction = float(cfg["backtest"].get("train_fraction", 0.7))
    X_train, X_test, y_train, y_test = train_test_split_time(features, labels, train_fraction)
    threshold = float(mcfg.get("probability_threshold", 0.55))
    random_state = int(mcfg.get("random_state", 42))
    requested = list(mcfg.get("algorithms", ["logistic_regression", "random_forest", "xgboost"]))

    catalog = _model_catalog(random_state)
    models: Dict[str, object] = {}
    for name in requested:
        if name in catalog:
            models[name] = catalog[name]
        elif name == "xgboost" and "xgboost_fallback_gradient_boosting" in catalog:
            models["xgboost_fallback_gradient_boosting"] = catalog["xgboost_fallback_gradient_boosting"]

    if not models:
        models = {"random_forest": catalog["random_forest"]}

    # If a time split creates a single-class training set, use a dummy baseline to keep the pipeline reviewable.
    if y_train.nunique() < 2:
        majority = int(y_train.mode().iloc[0]) if len(y_train) else 0
        models = {"dummy_majority_class": DummyClassifier(strategy="constant", constant=majority)}

    bundles: Dict[str, ModelBundle] = {}
    pred_frame = pd.DataFrame(index=X_test.index)
    pred_frame["actual"] = y_test
    metrics_rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_test)
            proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        else:
            proba = model.predict(X_test).astype(float)
        pred = (proba >= threshold).astype(int)
        metrics = _metrics(y_test, proba, pred)
        pred_frame[f"{name}_proba"] = proba
        pred_frame[f"{name}_accept"] = pred
        importance = _feature_importance(name, model)
        bundles[name] = ModelBundle(name, model, metrics, importance)
        metrics_rows.append({"model": name, **metrics})

    return bundles, pd.DataFrame(metrics_rows), pred_frame
