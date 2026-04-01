"""Tests for backtesting engine and metrics."""

import pytest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from start.backtest.engine import backtest_signals
from start.backtest.metrics import compute_metrics, compare_strategies


def _make_backtest_data(n=500):
    """Create synthetic data for backtesting."""
    np.random.seed(42)
    timestamps = pd.date_range("2024-06-01 09:30", periods=n, freq="5min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close + np.random.randn(n) * 0.1,
        "high": close + np.abs(np.random.randn(n) * 0.3),
        "low": close - np.abs(np.random.randn(n) * 0.3),
        "close": close,
        "volume": np.random.randint(1000, 50000, n),
    })


def test_backtest_buy_and_hold():
    df = _make_backtest_data()
    signals = pd.Series(1, index=df.index)  # Always long
    bt = backtest_signals(df, signals)

    assert "equity_curve" in bt
    assert "trades" in bt
    assert "initial_capital" in bt
    assert len(bt["equity_curve"]) == len(df)


def test_backtest_flat():
    df = _make_backtest_data()
    signals = pd.Series(0, index=df.index)  # Always flat
    bt = backtest_signals(df, signals)
    metrics = compute_metrics(bt)

    # Flat strategy should have zero return (no trades)
    assert metrics["n_trades"] == 0
    assert abs(metrics["total_return"]) < 1e-10


def test_backtest_alternating_signals():
    df = _make_backtest_data(100)
    # Alternate buy/sell every 10 bars
    signals = pd.Series(0, index=df.index)
    for i in range(0, 100, 20):
        signals.iloc[i:i+10] = 1

    bt = backtest_signals(df, signals)
    metrics = compute_metrics(bt)
    assert metrics["n_trades"] > 0


def test_metrics_keys():
    df = _make_backtest_data()
    signals = pd.Series(1, index=df.index)
    bt = backtest_signals(df, signals)
    metrics = compute_metrics(bt)

    required_keys = [
        "net_pnl", "total_return", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "n_trades", "win_rate", "volatility",
    ]
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"


def test_compare_strategies():
    results = {
        "strat_a": {"sharpe_ratio": 1.5, "total_return": 0.1},
        "strat_b": {"sharpe_ratio": 0.8, "total_return": 0.05},
    }
    df = compare_strategies(results)
    assert len(df) == 2
    # Should be sorted by Sharpe descending
    assert df.index[0] == "strat_a"


def test_drawdown_computation():
    df = _make_backtest_data()
    signals = pd.Series(1, index=df.index)
    bt = backtest_signals(df, signals)
    eq = bt["equity_curve"]

    # Drawdown should be non-negative
    assert (eq["drawdown"] >= 0).all()
    # Max drawdown should be ≤ 1
    assert eq["drawdown"].max() <= 1.0


def test_length_mismatch_raises():
    df = _make_backtest_data(100)
    signals = pd.Series(1, index=range(50))  # Wrong length
    with pytest.raises(ValueError):
        backtest_signals(df, signals)
