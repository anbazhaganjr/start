"""
Model analysis utilities.

Feature importance, statistical validation (t-test, ANOVA),
and model comparison tools.
"""

import numpy as np
import pandas as pd
from scipy import stats

from start.utils.logger import get_logger

logger = get_logger(__name__)


def feature_importance_analysis(
    model,
    feature_names: list[str],
    top_n: int = 15,
) -> pd.DataFrame:
    """
    Extract and rank feature importances from a trained model.

    Supports RandomForest (via feature_importances_) and
    Logistic/Ridge (via coefficient magnitudes).

    Args:
        model: Trained model (sklearn pipeline or wrapper).
        feature_names: List of feature column names.
        top_n: Number of top features to return.

    Returns:
        DataFrame with columns: feature, importance, rank.
    """
    # Extract the sklearn model from pipeline or wrapper
    if hasattr(model, "pipeline"):
        estimator = model.pipeline.named_steps.get("model")
    elif hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")
    else:
        estimator = model

    if estimator is None:
        logger.warning("[analysis] Could not extract estimator from model")
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    # Get importances based on model type
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = np.abs(estimator.coef_).flatten()
    else:
        logger.warning(f"[analysis] Model type {type(estimator)} has no importances")
        return pd.DataFrame(columns=["feature", "importance", "rank"])

    # Ensure lengths match
    n = min(len(feature_names), len(importances))
    df = pd.DataFrame({
        "feature": feature_names[:n],
        "importance": importances[:n],
    })

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    logger.info(f"[analysis] Top {min(top_n, len(df))} features:")
    for _, row in df.head(top_n).iterrows():
        logger.info(f"  {row['rank']:2d}. {row['feature']:<25s} {row['importance']:.6f}")

    return df.head(top_n)


def perform_t_test(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    strategy_a: str = "Strategy A",
    strategy_b: str = "Strategy B",
    alpha: float = 0.05,
) -> dict:
    """
    Welch's t-test comparing mean returns of two strategies.

    Tests H0: mean(returns_a) == mean(returns_b).

    Args:
        returns_a: Return series for strategy A.
        returns_b: Return series for strategy B.
        strategy_a: Name of strategy A.
        strategy_b: Name of strategy B.
        alpha: Significance level.

    Returns:
        Dict with test statistic, p-value, and conclusion.
    """
    t_stat, p_value = stats.ttest_ind(returns_a, returns_b, equal_var=False)

    significant = p_value < alpha
    conclusion = (
        f"SIGNIFICANT difference (p={p_value:.4f} < {alpha})"
        if significant
        else f"NO significant difference (p={p_value:.4f} >= {alpha})"
    )

    result = {
        "test": "Welch's t-test",
        "strategy_a": strategy_a,
        "strategy_b": strategy_b,
        "mean_a": float(np.mean(returns_a)),
        "mean_b": float(np.mean(returns_b)),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": significant,
        "conclusion": conclusion,
    }

    logger.info(
        f"[t-test] {strategy_a} vs {strategy_b}: "
        f"t={t_stat:.4f}, p={p_value:.4f} → {conclusion}"
    )

    return result


def perform_anova(
    strategy_returns: dict[str, np.ndarray],
    alpha: float = 0.05,
) -> dict:
    """
    One-way ANOVA comparing mean returns across multiple strategies.

    Tests H0: All strategy means are equal.

    Args:
        strategy_returns: Dict of {strategy_name: return_array}.
        alpha: Significance level.

    Returns:
        Dict with F-statistic, p-value, and per-strategy stats.
    """
    groups = list(strategy_returns.values())
    names = list(strategy_returns.keys())

    if len(groups) < 2:
        return {"test": "ANOVA", "error": "Need at least 2 strategies"}

    f_stat, p_value = stats.f_oneway(*groups)
    significant = p_value < alpha

    # Per-strategy descriptive stats
    group_stats = []
    for name, returns in strategy_returns.items():
        group_stats.append({
            "strategy": name,
            "n": len(returns),
            "mean": float(np.mean(returns)),
            "std": float(np.std(returns)),
            "median": float(np.median(returns)),
        })

    conclusion = (
        f"SIGNIFICANT difference among strategies (p={p_value:.4f} < {alpha})"
        if significant
        else f"NO significant difference (p={p_value:.4f} >= {alpha})"
    )

    result = {
        "test": "One-way ANOVA",
        "n_strategies": len(groups),
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "significant": significant,
        "conclusion": conclusion,
        "group_stats": group_stats,
    }

    logger.info(
        f"[ANOVA] {len(groups)} strategies: F={f_stat:.4f}, p={p_value:.4f} → {conclusion}"
    )

    return result


def correlation_analysis(
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """
    Compute feature correlation matrix.

    Args:
        df: Feature DataFrame.
        feature_cols: Feature columns to analyze.

    Returns:
        Correlation matrix DataFrame.
    """
    available = [c for c in feature_cols if c in df.columns]
    corr = df[available].corr()

    # Log highly correlated pairs
    high_corr = []
    for i in range(len(corr)):
        for j in range(i + 1, len(corr)):
            val = abs(corr.iloc[i, j])
            if val > 0.8:
                high_corr.append((corr.index[i], corr.columns[j], val))

    if high_corr:
        logger.info(f"[analysis] High correlations (>0.8):")
        for f1, f2, val in sorted(high_corr, key=lambda x: -x[2]):
            logger.info(f"  {f1} ↔ {f2}: {val:.3f}")

    return corr
