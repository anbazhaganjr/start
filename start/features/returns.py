"""
Return and volatility computations.

All computed per-symbol with proper session boundary handling
(no overnight contamination for intraday data).
"""

import numpy as np
import pandas as pd

from start.utils.constants import ROLLING_VOL_WINDOW
from start.utils.logger import get_logger

logger = get_logger(__name__)


def add_returns(
    df: pd.DataFrame,
    rolling_window: int = ROLLING_VOL_WINDOW,
) -> pd.DataFrame:
    """
    Add return and volatility features to a DataFrame.

    Adds columns:
        - simple_return:      (close[t] - close[t-1]) / close[t-1]
        - log_return:         ln(close[t] / close[t-1])
        - rolling_volatility: rolling std of log returns (N-bar window)
        - cumulative_return:  cumulative sum of log returns from first bar
        - intraday_range:     (high - low) / close (normalized bar range)
        - volume_ma_20:       20-bar moving average of volume
        - volume_ratio:       volume / volume_ma_20 (relative volume)

    Args:
        df: DataFrame with [close, high, low, volume] columns.
        rolling_window: Window for rolling volatility.

    Returns:
        DataFrame with return columns appended.
    """
    df = df.copy()

    if "symbol" in df.columns:
        groups = []
        for symbol, group in df.groupby("symbol"):
            group = _compute_returns(group, rolling_window)
            groups.append(group)
        df = pd.concat(groups, ignore_index=True)
    else:
        df = _compute_returns(df, rolling_window)

    return df


def _compute_returns(df: pd.DataFrame, rolling_window: int) -> pd.DataFrame:
    """Compute return features for a single symbol."""
    close = df["close"].astype(float)

    # Simple return
    df["simple_return"] = close.pct_change()

    # Log return (additive, better for aggregation)
    df["log_return"] = np.log(close / close.shift(1))

    # Rolling volatility (std of log returns over window)
    df["rolling_volatility"] = (
        df["log_return"]
        .rolling(window=rolling_window, min_periods=rolling_window)
        .std()
    )

    # Cumulative return from start
    df["cumulative_return"] = df["log_return"].cumsum()

    # Intraday range (normalized)
    df["intraday_range"] = (df["high"] - df["low"]) / close

    # Volume features
    df["volume_ma_20"] = df["volume"].rolling(window=20, min_periods=1).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma_20"].replace(0, np.nan)

    return df
