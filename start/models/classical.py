"""
Classical ML models for next-bar direction prediction.

All models follow a consistent interface:
    - fit(X_train, y_train) → self
    - predict(X_test) → array of 0/1
    - predict_proba(X_test) → array of probabilities
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from start.utils.logger import get_logger

logger = get_logger(__name__)


class LogisticModel:
    """Logistic regression for direction classification."""

    name = "logistic"

    def __init__(self, C: float = 1.0, max_iter: int = 1000):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=C, max_iter=max_iter, random_state=42)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


class RidgeModel:
    """
    Ridge regression for return magnitude prediction.

    Predicts continuous return, then converts to direction signal.
    """

    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=alpha)),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RidgeModel":
        # Convert binary target to return for regression
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.pipeline.predict(X)
        return (preds > 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.pipeline.predict(X), 0, 1)


class RandomForestModel:
    """Random Forest classifier for direction prediction."""

    name = "random_forest"

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 10,
        min_samples_leaf: int = 20,
    ):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1,
            )),
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        rf = self.pipeline.named_steps["model"]
        importance = pd.DataFrame({
            "feature": feature_names,
            "importance": rf.feature_importances_,
        }).sort_values("importance", ascending=False)
        return importance


def get_model(name: str, **kwargs):
    """Factory to create model by name."""
    models = {
        "logistic": LogisticModel,
        "ridge": RidgeModel,
        "random_forest": RandomForestModel,
    }
    if name not in models:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(models.keys())}")
    return models[name](**kwargs)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "",
) -> dict:
    """Compute classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "accuracy": acc,
        "precision_up": report.get("1", {}).get("precision", 0),
        "recall_up": report.get("1", {}).get("recall", 0),
        "f1_up": report.get("1", {}).get("f1-score", 0),
        "precision_down": report.get("0", {}).get("precision", 0),
        "recall_down": report.get("0", {}).get("recall", 0),
        "f1_down": report.get("0", {}).get("f1-score", 0),
    }

    if model_name:
        logger.info(
            f"[{model_name}] Accuracy: {acc:.4f}, "
            f"F1 up: {metrics['f1_up']:.4f}, F1 down: {metrics['f1_down']:.4f}"
        )

    return metrics
