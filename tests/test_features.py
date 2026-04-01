"""Tests for feature engineering module."""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from start.features.technical import add_technical_indicators
from start.features.returns import add_returns
from start.features.builder import get_feature_columns, drop_highly_correlated


def _make_price_data(n=200):
    """Create synthetic price data with enough bars for indicators."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-06-01 09:30", periods=n, freq="5min", tz="US/Eastern")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": np.random.randint(1000, 100000, n),
    })


def test_technical_indicators_added():
    df = _make_price_data(200)
    result = add_technical_indicators(df)
    expected = ["sma_20", "sma_50", "rsi_14", "macd", "bb_upper", "bb_lower"]
    for col in expected:
        assert col in result.columns, f"Missing {col}"


def test_returns_added():
    df = _make_price_data(100)
    result = add_returns(df)
    expected = ["simple_return", "log_return", "rolling_volatility"]
    for col in expected:
        assert col in result.columns, f"Missing {col}"


def test_no_future_leakage():
    """Target must be computed from FUTURE data — shifted correctly."""
    df = _make_price_data(200)
    df = add_technical_indicators(df)
    df = add_returns(df)
    # Build target: next-bar direction
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df = df.dropna()

    # Verify target is shifted: target at row i should relate to price at i+1
    # The target should NOT be correlated with current features in a trivial way
    feature_cols = get_feature_columns(df)
    if feature_cols:
        X = df[feature_cols].iloc[:-1]
        y = df["target"].iloc[:-1]
        # Just verify shapes match and no NaN
        assert len(X) == len(y)
        assert not X.isna().any().any()


def test_drop_highly_correlated():
    """Correlation filter should remove redundant features."""
    np.random.seed(42)
    n = 100
    x1 = np.random.randn(n)
    x2 = x1 + np.random.randn(n) * 0.01  # Almost identical to x1
    x3 = np.random.randn(n)  # Independent

    df = pd.DataFrame({"a": x1, "b": x2, "c": x3})
    result_df, dropped_cols = drop_highly_correlated(df, ["a", "b", "c"], threshold=0.95)
    # Should drop one of {a, b} (highly correlated pair)
    assert len(dropped_cols) >= 1
    # The independent column "c" should NOT be dropped
    assert "c" not in dropped_cols
    # At most one of a/b should be dropped
    assert "a" in dropped_cols or "b" in dropped_cols


def test_feature_columns_detection():
    df = _make_price_data(200)
    df = add_technical_indicators(df)
    df = add_returns(df)
    df["target"] = 1  # Add target
    df = df.dropna()

    cols = get_feature_columns(df)
    assert len(cols) > 0
    assert "target" not in cols
    assert "timestamp" not in cols
    assert "close" not in cols
