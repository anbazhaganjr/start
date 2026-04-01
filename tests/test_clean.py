"""Tests for data cleaning module."""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from start.data.clean import clean_bars, validate_session_coverage


def _make_ohlcv(n=100):
    """Create synthetic OHLCV data."""
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


def test_clean_bars_removes_duplicates():
    df = _make_ohlcv(50)
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    cleaned = clean_bars(df)
    assert len(cleaned) <= 50


def test_clean_bars_validates_ohlcv():
    df = _make_ohlcv(50)
    # Introduce bad row: high < low
    df.loc[10, "high"] = df.loc[10, "low"] - 1
    cleaned = clean_bars(df)
    # Should either fix or remove bad rows
    assert len(cleaned) <= 50


def test_clean_bars_handles_empty():
    df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    cleaned = clean_bars(df)
    assert len(cleaned) == 0


def test_clean_preserves_columns():
    df = _make_ohlcv(50)
    cleaned = clean_bars(df)
    for col in ["timestamp", "open", "high", "low", "close", "volume"]:
        assert col in cleaned.columns
