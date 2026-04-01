"""
Baseline trading strategies.

These serve as benchmarks for evaluating ML/RL models.
Each strategy produces a signal series: 1 = long, 0 = flat, -1 = short.
"""

import numpy as np
import pandas as pd

from start.utils.logger import get_logger

logger = get_logger(__name__)


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """
    Buy-and-hold: always long.

    Args:
        df: Feature DataFrame with at least [timestamp, close].

    Returns:
        Series of signals (all 1s).
    """
    signals = pd.Series(1, index=df.index, name="signal")
    logger.info(f"[baseline] Buy-and-hold: {len(signals)} bars, always long")
    return signals


def ma_crossover(
    df: pd.DataFrame,
    short_col: str = "sma_20",
    long_col: str = "sma_50",
) -> pd.Series:
    """
    MA crossover: long when short MA > long MA, flat otherwise.

    If SMA columns aren't present, computes them from close.

    Args:
        df: Feature DataFrame with close and optionally SMA columns.
        short_col: Column name for short-period MA.
        long_col: Column name for long-period MA.

    Returns:
        Series of signals (1 = long, 0 = flat).
    """
    df = df.copy()

    if short_col not in df.columns:
        df[short_col] = df["close"].rolling(20).mean()
    if long_col not in df.columns:
        df[long_col] = df["close"].rolling(50).mean()

    signals = (df[short_col] > df[long_col]).astype(int)
    signals.name = "signal"

    n_long = (signals == 1).sum()
    n_flat = (signals == 0).sum()
    logger.info(
        f"[baseline] MA crossover: {n_long} long ({n_long/len(signals):.1%}), "
        f"{n_flat} flat ({n_flat/len(signals):.1%})"
    )

    return signals


def rsi_mean_reversion(
    df: pd.DataFrame,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.Series:
    """
    RSI mean-reversion: long when RSI < oversold, flat when RSI > overbought.

    Args:
        df: Feature DataFrame with rsi_14 column.
        oversold: RSI threshold to go long.
        overbought: RSI threshold to go flat.

    Returns:
        Series of signals.
    """
    signals = pd.Series(0, index=df.index, name="signal")

    if "rsi_14" not in df.columns:
        logger.warning("[baseline] RSI column not found, returning flat")
        return signals

    # State machine: track current position
    position = 0
    for i in range(len(df)):
        rsi = df["rsi_14"].iloc[i]
        if np.isnan(rsi):
            signals.iloc[i] = position
            continue

        if rsi < oversold:
            position = 1  # Go long
        elif rsi > overbought:
            position = 0  # Go flat

        signals.iloc[i] = position

    n_long = (signals == 1).sum()
    logger.info(
        f"[baseline] RSI mean-reversion: {n_long} long ({n_long/len(signals):.1%})"
    )

    return signals
