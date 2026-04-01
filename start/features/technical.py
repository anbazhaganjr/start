"""
Technical indicator computation.

Computes MACD, RSI, Bollinger Bands, SMA(20/50), and VWAP distance.
Uses ta-lib when available, falls back to pure pandas for robustness.
"""

from typing import Optional

import numpy as np
import pandas as pd

from start.utils.constants import (
    MACD_FAST, MACD_SLOW, MACD_SIGNAL,
    RSI_PERIOD, SMA_SHORT, SMA_LONG,
    BOLLINGER_PERIOD, BOLLINGER_STD,
)
from start.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import ta-lib; fall back gracefully
try:
    import talib
    _HAS_TALIB = True
    logger.info("Using ta-lib for technical indicators")
except ImportError:
    _HAS_TALIB = False
    logger.info("ta-lib not available, using pandas fallback")


# ---------------------------------------------------------------------------
# Pure-pandas implementations (always available)
# ---------------------------------------------------------------------------
def _ema_pandas(series: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=False).mean()


def _sma_pandas(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=period, min_periods=period).mean()


def _rsi_pandas(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """RSI via pandas — Wilder's smoothing method."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


def _macd_pandas(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram via pandas."""
    ema_fast = _ema_pandas(close, fast)
    ema_slow = _ema_pandas(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema_pandas(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_pandas(
    close: pd.Series,
    period: int = BOLLINGER_PERIOD,
    num_std: float = BOLLINGER_STD,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands (upper, middle, lower)."""
    middle = _sma_pandas(close, period)
    rolling_std = close.rolling(window=period, min_periods=period).std()
    upper = middle + (rolling_std * num_std)
    lower = middle - (rolling_std * num_std)
    return upper, middle, lower


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def add_technical_indicators(
    df: pd.DataFrame,
    use_talib: bool = True,
) -> pd.DataFrame:
    """
    Add all technical indicators to a DataFrame.

    Input must have columns: [close, high, low, volume] and optionally [vwap].
    Computed per-symbol if 'symbol' column exists.

    Adds columns:
        - sma_20, sma_50
        - rsi_14
        - macd, macd_signal, macd_hist
        - bb_upper, bb_middle, bb_lower, bb_width, bb_pct
        - vwap_distance (if vwap column present)

    Args:
        df: OHLCV DataFrame.
        use_talib: Try ta-lib first (falls back to pandas if unavailable).

    Returns:
        DataFrame with indicator columns appended.
    """
    df = df.copy()

    if "symbol" in df.columns:
        # Process each symbol separately to avoid cross-contamination
        groups = []
        for symbol, group in df.groupby("symbol"):
            group = _compute_indicators(group, use_talib)
            groups.append(group)
        df = pd.concat(groups, ignore_index=True)
    else:
        df = _compute_indicators(df, use_talib)

    return df


def _compute_indicators(df: pd.DataFrame, use_talib: bool) -> pd.DataFrame:
    """Compute indicators for a single symbol's data."""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # --- SMA ---
    df["sma_20"] = _sma_pandas(close, SMA_SHORT)
    df["sma_50"] = _sma_pandas(close, SMA_LONG)

    # --- RSI ---
    if use_talib and _HAS_TALIB:
        df["rsi_14"] = talib.RSI(close.values, timeperiod=RSI_PERIOD)
    else:
        df["rsi_14"] = _rsi_pandas(close, RSI_PERIOD)

    # --- MACD ---
    if use_talib and _HAS_TALIB:
        macd, signal, hist = talib.MACD(
            close.values,
            fastperiod=MACD_FAST,
            slowperiod=MACD_SLOW,
            signalperiod=MACD_SIGNAL,
        )
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist
    else:
        macd, signal, hist = _macd_pandas(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist

    # --- Bollinger Bands ---
    if use_talib and _HAS_TALIB:
        upper, middle, lower = talib.BBANDS(
            close.values,
            timeperiod=BOLLINGER_PERIOD,
            nbdevup=BOLLINGER_STD,
            nbdevdn=BOLLINGER_STD,
        )
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
    else:
        upper, middle, lower = _bollinger_pandas(close, BOLLINGER_PERIOD, BOLLINGER_STD)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower

    # Bollinger Band width and %B
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_middle"]
    df["bb_pct"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    # --- VWAP distance ---
    if "vwap" in df.columns:
        df["vwap_distance"] = (close - df["vwap"]) / df["vwap"]

    return df
