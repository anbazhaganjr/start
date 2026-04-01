"""
Walk-forward validation engine.

Implements time-series aware training with expanding and sliding windows.
Never shuffles data — respects temporal ordering to prevent leakage.
"""

from typing import Optional
import time as time_module

import numpy as np
import pandas as pd

from start.features.builder import get_feature_columns, get_X_y
from start.models.classical import evaluate_predictions
from start.utils.logger import get_logger

logger = get_logger(__name__)


def walk_forward_train(
    df: pd.DataFrame,
    model_factory,
    mode: str = "expanding",
    train_bars: int = 1500,
    test_bars: int = 200,
    step_bars: int = 200,
    feature_cols: Optional[list[str]] = None,
) -> dict:
    """
    Walk-forward validation with expanding or sliding windows.

    Time-ordered: train on [0, t], test on [t, t+W], advance by step.
    No shuffling, no future data leakage.

    Args:
        df: Feature DataFrame with target column, sorted by timestamp.
        model_factory: Callable that returns a fresh model instance.
        mode: "expanding" (growing train window) or "sliding" (fixed-size).
        train_bars: Initial training window size.
        test_bars: Test window size per fold.
        step_bars: Step size between folds.
        feature_cols: Feature columns to use. Defaults to auto-detect.

    Returns:
        Dict with:
            - predictions: DataFrame of all out-of-sample predictions
            - fold_metrics: List of per-fold metric dicts
            - overall_metrics: Aggregated metrics across all folds
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X_full, y_full = get_X_y(df, feature_cols)
    n = len(X_full)

    all_preds = []
    all_actuals = []
    all_indices = []
    all_probas = []
    fold_metrics = []

    fold = 0
    train_end = train_bars

    while train_end + test_bars <= n:
        if mode == "expanding":
            train_start = 0
        else:  # sliding
            train_start = max(0, train_end - train_bars)

        test_end = min(train_end + test_bars, n)

        X_train = X_full.iloc[train_start:train_end]
        y_train = y_full.iloc[train_start:train_end]
        X_test = X_full.iloc[train_end:test_end]
        y_test = y_full.iloc[train_end:test_end]

        # Train
        model = model_factory()
        t0 = time_module.time()
        model.fit(X_train, y_train)
        train_time = time_module.time() - t0

        # Predict
        preds = model.predict(X_test)
        probas = model.predict_proba(X_test) if hasattr(model, "predict_proba") else preds.astype(float)

        # Evaluate this fold
        fold_metric = evaluate_predictions(
            y_test.values, preds, model_name=f"fold_{fold}"
        )
        fold_metric["fold"] = fold
        fold_metric["train_size"] = len(X_train)
        fold_metric["test_size"] = len(X_test)
        fold_metric["train_time"] = train_time
        fold_metrics.append(fold_metric)

        # Collect predictions
        all_preds.extend(preds)
        all_actuals.extend(y_test.values)
        all_indices.extend(y_test.index.tolist())
        all_probas.extend(probas)

        fold += 1
        train_end += step_bars

    if not all_preds:
        logger.warning("[training] No folds executed — insufficient data")
        return {"predictions": pd.DataFrame(), "fold_metrics": [], "overall_metrics": {}}

    # Aggregate predictions
    predictions_df = pd.DataFrame({
        "idx": all_indices,
        "actual": all_actuals,
        "predicted": all_preds,
        "probability": all_probas,
    })

    # Overall metrics
    overall = evaluate_predictions(
        np.array(all_actuals),
        np.array(all_preds),
        model_name="overall",
    )
    overall["n_folds"] = fold
    overall["total_predictions"] = len(all_preds)

    # Mean fold metrics
    fold_df = pd.DataFrame(fold_metrics)
    overall["mean_fold_accuracy"] = fold_df["accuracy"].mean()
    overall["std_fold_accuracy"] = fold_df["accuracy"].std()

    logger.info(
        f"[training] Walk-forward complete: {fold} folds, "
        f"{len(all_preds)} predictions, "
        f"accuracy: {overall['accuracy']:.4f} ± {overall['std_fold_accuracy']:.4f}"
    )

    return {
        "predictions": predictions_df,
        "fold_metrics": fold_metrics,
        "overall_metrics": overall,
    }


def train_all_models(
    df: pd.DataFrame,
    model_configs: Optional[dict] = None,
    mode: str = "expanding",
    train_bars: int = 1500,
    test_bars: int = 200,
    step_bars: int = 200,
) -> dict:
    """
    Train all configured models via walk-forward validation.

    Args:
        df: Feature DataFrame for a single symbol.
        model_configs: Dict of {name: factory_callable}. Defaults to all models.
        mode: Walk-forward mode.
        train_bars: Initial training window.
        test_bars: Test window per fold.
        step_bars: Step between folds.

    Returns:
        Dict of {model_name: walk_forward_results}.
    """
    from start.models.classical import LogisticModel, RidgeModel, RandomForestModel
    from start.models.lstm import LSTMModel
    from start.models.cnn import CNNModel

    if model_configs is None:
        from config import get_setting
        mc = get_setting("models", default={})
        rf = mc.get("random_forest", {})
        lstm_cfg = mc.get("lstm", {})
        cnn_cfg = mc.get("cnn", {})

        model_configs = {
            "logistic": lambda: LogisticModel(
                C=mc.get("logistic", {}).get("C", 1.0),
                max_iter=mc.get("logistic", {}).get("max_iter", 1000),
            ),
            "ridge": lambda: RidgeModel(
                alpha=mc.get("ridge", {}).get("alpha", 1.0),
            ),
            "random_forest": lambda: RandomForestModel(
                n_estimators=rf.get("n_estimators", 50),
                max_depth=rf.get("max_depth", 8),
                min_samples_leaf=rf.get("min_samples_leaf", 20),
            ),
            "lstm": lambda: LSTMModel(
                hidden_size=lstm_cfg.get("hidden_size", 32),
                seq_len=lstm_cfg.get("seq_len", 20),
                max_epochs=lstm_cfg.get("max_epochs", 30),
                patience=lstm_cfg.get("patience", 5),
                lr=lstm_cfg.get("learning_rate", 1e-3),
            ),
            "cnn": lambda: CNNModel(
                seq_len=cnn_cfg.get("seq_len", 20),
                max_epochs=cnn_cfg.get("max_epochs", 30),
                patience=cnn_cfg.get("patience", 5),
                lr=cnn_cfg.get("learning_rate", 1e-3),
            ),
        }

    results = {}
    feature_cols = get_feature_columns(df)

    for name, factory in model_configs.items():
        logger.info(f"\n{'='*40}")
        logger.info(f"Training: {name}")
        logger.info(f"{'='*40}")

        t0 = time_module.time()
        result = walk_forward_train(
            df=df,
            model_factory=factory,
            mode=mode,
            train_bars=train_bars,
            test_bars=test_bars,
            step_bars=step_bars,
            feature_cols=feature_cols,
        )
        total_time = time_module.time() - t0

        result["overall_metrics"]["total_time"] = total_time
        result["overall_metrics"]["model_name"] = name
        results[name] = result

        logger.info(f"[{name}] Total time: {total_time:.1f}s")

    return results
