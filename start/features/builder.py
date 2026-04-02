"""
Feature matrix builder.

Assembles all features, creates target variable, enforces causal alignment,
and optionally removes highly correlated features.
"""

from typing import Optional

import numpy as np
import pandas as pd

from start.features.technical import add_technical_indicators
from start.features.returns import add_returns
from start.utils.constants import SMA_LONG
from start.utils.logger import get_logger

logger = get_logger(__name__)

# Features that are inputs to models (not metadata)
FEATURE_COLUMNS = [
    # Technical indicators
    "sma_20", "sma_50",
    "rsi_14",
    "macd", "macd_signal", "macd_hist",
    "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct",
    # Returns & volatility
    "simple_return", "log_return",
    "rolling_volatility", "cumulative_return",
    "intraday_range",
    "volume_ratio",
]

# Metadata columns to preserve but not use as features
META_COLUMNS = [
    "timestamp", "date", "symbol", "provider",
    "open", "high", "low", "close", "volume",
]


def build_features(
    df: pd.DataFrame,
    drop_warmup: bool = True,
    add_target: bool = True,
    drop_correlated: bool = False,
    corr_threshold: float = 0.95,
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps:
        1. Add technical indicators (MACD, RSI, Bollinger, SMA)
        2. Add return/volatility features
        3. Drop warmup rows (first SMA_LONG bars per symbol)
        4. Create target variable (next-bar direction)
        5. Drop highly correlated features (optional)
        6. Drop rows with NaN in feature columns

    Args:
        df: Cleaned OHLCV DataFrame with [timestamp, open, high, low, close, volume, symbol].
        drop_warmup: Remove first N bars where SMA_LONG is NaN.
        add_target: Add 'target' column (1 if next close > current close, else 0).
        drop_correlated: Remove features with correlation > threshold.
        corr_threshold: Correlation threshold for removal.

    Returns:
        Feature matrix DataFrame.
    """
    logger.info(f"[builder] Starting feature build: {len(df)} input rows")

    # Step 1: Technical indicators
    df = add_technical_indicators(df)
    logger.info(f"[builder] Added technical indicators")

    # Step 2: Returns and volatility
    df = add_returns(df)
    logger.info(f"[builder] Added return features")

    # Step 3: Drop warmup period (SMA_LONG bars at start of each symbol)
    if drop_warmup and "symbol" in df.columns:
        before = len(df)
        groups = []
        for symbol, group in df.groupby("symbol"):
            # Drop rows where SMA_LONG hasn't warmed up
            group = group.iloc[SMA_LONG:]
            groups.append(group)
        df = pd.concat(groups, ignore_index=True)
        logger.info(f"[builder] Dropped {before - len(df)} warmup rows")

    # Step 4: Create target (next-bar direction)
    if add_target:
        df = _add_target(df)

    # Step 5: Drop NaN rows in feature columns
    available_features = [c for c in FEATURE_COLUMNS if c in df.columns]
    before = len(df)
    df = df.dropna(subset=available_features)
    if len(df) < before:
        logger.info(f"[builder] Dropped {before - len(df)} rows with NaN features")

    # Step 6: Drop highly correlated features
    if drop_correlated:
        df, dropped = drop_highly_correlated(df, available_features, corr_threshold)
        if dropped:
            logger.info(
                f"[builder] Dropped {len(dropped)} correlated features: {dropped}"
            )

    logger.info(f"[builder] Final: {len(df)} rows, {len(get_feature_columns(df))} features")

    return df


def _add_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add binary target: 1 if next bar's close > current close, else 0.

    CAUSAL: target[t] uses close[t+1], which is the NEXT bar.
    The last bar of each symbol's data will have NaN target (dropped later).
    """
    df = df.copy()

    if "symbol" in df.columns:
        # Shift within each symbol to avoid cross-symbol leakage
        df["target"] = df.groupby("symbol")["close"].shift(-1)
        df["target"] = (df["target"] > df["close"]).astype(float)
        # Last bar per symbol has NaN target
        df.loc[df.groupby("symbol").tail(1).index, "target"] = np.nan
    else:
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
        df.iloc[-1, df.columns.get_loc("target")] = np.nan

    df = df.dropna(subset=["target"])
    df["target"] = df["target"].astype(int)

    logger.info(
        f"[builder] Target distribution: "
        f"{(df['target'] == 1).sum()} up ({(df['target'] == 1).mean():.1%}), "
        f"{(df['target'] == 0).sum()} down ({(df['target'] == 0).mean():.1%})"
    )

    return df


def drop_highly_correlated(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove features with absolute correlation above threshold.

    Keeps the first feature in each correlated pair (preserves the one
    that appears earlier in FEATURE_COLUMNS).

    Returns:
        Tuple of (filtered DataFrame, list of dropped column names).
    """
    corr_matrix = df[feature_cols].corr().abs()

    # Get upper triangle (avoid double counting)
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find columns to drop
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]

    if to_drop:
        df = df.drop(columns=to_drop)

    return df, to_drop


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature columns actually present in the DataFrame."""
    return [c for c in df.columns if c in FEATURE_COLUMNS]


def get_X_y(
    df: pd.DataFrame,
    feature_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and target vector y from a feature DataFrame.

    Args:
        df: Feature DataFrame with target column.
        feature_cols: Specific features to use. Defaults to all available.

    Returns:
        (X, y) tuple.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    X = df[feature_cols].copy()
    y = df["target"].copy()

    return X, y


def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate summary statistics for the feature dataset.

    Returns a DataFrame with per-symbol statistics.
    """
    feature_cols = get_feature_columns(df)
    price_cols = ["open", "high", "low", "close", "volume"]
    all_cols = [c for c in price_cols + feature_cols if c in df.columns]

    if "symbol" in df.columns:
        summary = df.groupby("symbol")[all_cols].agg(
            ["mean", "std", "min", "max", "median"]
        )
    else:
        summary = df[all_cols].agg(
            ["mean", "std", "min", "max", "median"]
        )

    return summary
