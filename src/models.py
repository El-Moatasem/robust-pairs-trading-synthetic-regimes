from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _build_model(name: str, random_state: int = 42):
    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=10,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=150,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=random_state,
            )
        except Exception:
            # Graceful fallback keeps the pipeline runnable if xgboost is unavailable.
            return RandomForestClassifier(n_estimators=150, max_depth=4, random_state=random_state)
    raise ValueError(f"Unknown model name: {name}")


def time_ordered_train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.30):
    n = len(X)
    split = int(n * (1 - test_size))
    split = max(1, min(split, n - 1))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def train_trade_filters(
    X: pd.DataFrame,
    y: pd.Series,
    model_names: list[str] | None = None,
    test_size: float = 0.30,
    random_state: int = 42,
) -> dict:
    """Train ML models and return fitted models plus evaluation metrics."""
    if model_names is None:
        model_names = ["logistic_regression", "random_forest", "xgboost"]
    if len(X) < 10 or y.nunique() < 2:
        return {"warning": "Not enough labeled data with both classes to train ML filters."}
    X_train, X_test, y_train, y_test = time_ordered_train_test_split(X, y, test_size)
    results = {}
    for name in model_names:
        model = _build_model(name, random_state)
        model.fit(X_train, y_train)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)
        pred = (prob >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "auc": float(roc_auc_score(y_test, prob)) if y_test.nunique() > 1 else np.nan,
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        results[name] = {"model": model, "metrics": metrics, "X_test_index": X_test.index, "probabilities": pd.Series(prob, index=X_test.index)}
    return results


def select_best_model(results: dict) -> tuple[str | None, object | None]:
    best_name, best_score, best_model = None, -np.inf, None
    for name, payload in results.items():
        if not isinstance(payload, dict) or "metrics" not in payload:
            continue
        score = payload["metrics"].get("auc")
        if score is None or np.isnan(score):
            score = payload["metrics"].get("f1", -np.inf)
        if score > best_score:
            best_name, best_score, best_model = name, score, payload.get("model")
    return best_name, best_model
