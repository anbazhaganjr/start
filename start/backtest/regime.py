"""
Market regime classifier.

Uses SPY as reference to classify each bar into bull/bear/sideways regime.
Enables regime-conditional performance analysis.
"""

import numpy as np
import pandas as pd

from start.utils.logger import get_logger

logger = get_logger(__name__)


def classify_regime(
    spy_df: pd.DataFrame,
    sma_period: int = 50,
    slope_window: int = 20,
) -> pd.Series:
    """
    Classify market regime using SPY price data.

    Rules:
        - Bull:     SMA slope > 0 AND price > SMA
        - Bear:     SMA slope < 0 AND price < SMA
        - Sideways: Neither condition met

    Args:
        spy_df: SPY DataFrame with [timestamp, close] columns.
        sma_period: SMA period for trend detection.
        slope_window: Window for computing SMA slope.

    Returns:
        Series of regime labels ("bull", "bear", "sideways") indexed like input.
    """
    df = spy_df.copy()
    close = df["close"].astype(float)

    # Compute SMA
    sma = close.rolling(window=sma_period, min_periods=sma_period).mean()

    # Compute SMA slope (change over slope_window)
    sma_slope = sma.diff(slope_window)

    # Classify
    regime = pd.Series("sideways", index=df.index)
    regime[(sma_slope > 0) & (close > sma)] = "bull"
    regime[(sma_slope < 0) & (close < sma)] = "bear"

    # Fill NaN period at start
    regime.iloc[:sma_period] = "sideways"

    # Stats
    counts = regime.value_counts()
    logger.info(
        f"[regime] Classification: "
        f"bull={counts.get('bull', 0)} ({counts.get('bull', 0)/len(regime):.1%}), "
        f"bear={counts.get('bear', 0)} ({counts.get('bear', 0)/len(regime):.1%}), "
        f"sideways={counts.get('sideways', 0)} ({counts.get('sideways', 0)/len(regime):.1%})"
    )

    return regime


def add_regime_to_features(
    features_df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add regime labels to a feature DataFrame by aligning on timestamp.

    Args:
        features_df: Feature DataFrame with timestamp column.
        spy_df: SPY DataFrame with timestamp and close.

    Returns:
        Feature DataFrame with 'regime' column added.
    """
    # Classify SPY
    regime = classify_regime(spy_df)

    # Create mapping from timestamp → regime
    spy_regime = pd.DataFrame({
        "timestamp": spy_df["timestamp"],
        "regime": regime.values,
    })

    # Merge on timestamp
    features_df = features_df.merge(
        spy_regime,
        on="timestamp",
        how="left",
    )

    # Fill missing regimes
    features_df["regime"] = features_df["regime"].fillna("sideways")

    return features_df


def metrics_by_regime(
    equity_curve: pd.DataFrame,
    regime_series: pd.Series,
) -> pd.DataFrame:
    """
    Compute performance metrics segmented by market regime.

    Args:
        equity_curve: DataFrame with [timestamp, equity] columns.
        regime_series: Series of regime labels aligned with equity curve.

    Returns:
        DataFrame with metrics per regime.
    """
    from start.backtest.metrics import compute_metrics

    results = []
    equity_curve = equity_curve.copy()
    equity_curve["regime"] = regime_series.values[:len(equity_curve)]

    for regime in ["bull", "bear", "sideways"]:
        mask = equity_curve["regime"] == regime
        if mask.sum() < 10:
            continue

        regime_eq = equity_curve[mask].copy()
        returns = regime_eq["equity"].pct_change().dropna()

        if len(returns) == 0:
            continue

        sharpe = (returns.mean() / returns.std()) * np.sqrt(252 * 7) if returns.std() > 0 else 0
        max_dd = regime_eq["drawdown"].max()
        mean_return = returns.mean()

        results.append({
            "regime": regime,
            "n_bars": int(mask.sum()),
            "mean_return": mean_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "volatility": returns.std(),
        })

    return pd.DataFrame(results)
