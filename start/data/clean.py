"""
Data cleaning and validation.

Ensures all bars meet quality standards before downstream use:
- Regular trading hours only (9:30–16:00 ET)
- OHLCV consistency checks
- Gap detection and forward-fill for small gaps
- Outlier flagging
- Deduplication
"""

from typing import Optional

import numpy as np
import pandas as pd

from start.utils.constants import MARKET_OPEN, MARKET_CLOSE, BARS_PER_DAY
from start.utils.logger import get_logger

logger = get_logger(__name__)


def clean_bars(
    df: pd.DataFrame,
    interval: str = "1h",
    max_gap_fill: int = 2,
    spike_sigma: float = 8.0,
) -> pd.DataFrame:
    """
    Clean and validate OHLCV bar data for a single symbol.

    Steps:
        1. Filter to regular trading hours (9:30 AM – 4:00 PM ET)
        2. Validate OHLCV consistency (low <= open/close <= high, volume >= 0)
        3. Remove duplicates on timestamp
        4. Detect and forward-fill small gaps (up to max_gap_fill bars)
        5. Flag price spikes > spike_sigma standard deviations
        6. Sort by timestamp

    Args:
        df: Raw DataFrame with [timestamp, open, high, low, close, volume, symbol].
        interval: Bar interval for gap detection.
        max_gap_fill: Max consecutive missing bars to forward-fill.
        spike_sigma: Z-score threshold for outlier flagging.

    Returns:
        Cleaned DataFrame with added columns [date, bar_of_day, is_outlier].
    """
    if df.empty:
        return df

    df = df.copy()

    # --- Ensure correct dtypes ---
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

    initial_count = len(df)

    # --- 1. Filter to regular trading hours ---
    df["_time"] = df["timestamp"].dt.strftime("%H:%M")
    df = df[(df["_time"] >= MARKET_OPEN) & (df["_time"] < MARKET_CLOSE)]
    df = df.drop(columns=["_time"])

    hours_filtered = initial_count - len(df)
    if hours_filtered > 0:
        logger.info(f"[clean] Removed {hours_filtered} pre/post-market bars")

    # --- 2. Deduplication ---
    dupes = df.duplicated(subset=["timestamp"], keep="first").sum()
    if dupes > 0:
        logger.info(f"[clean] Removed {dupes} duplicate timestamps")
        df = df.drop_duplicates(subset=["timestamp"], keep="first")

    # --- 3. OHLCV consistency ---
    # Fix: ensure high >= max(open, close) and low <= min(open, close)
    bad_high = df["high"] < df[["open", "close"]].max(axis=1)
    bad_low = df["low"] > df[["open", "close"]].min(axis=1)
    bad_volume = df["volume"] < 0

    n_fixed = bad_high.sum() + bad_low.sum()
    if n_fixed > 0:
        logger.warning(f"[clean] Fixed {n_fixed} OHLCV inconsistencies")
        df.loc[bad_high, "high"] = df.loc[bad_high, ["open", "close"]].max(axis=1)
        df.loc[bad_low, "low"] = df.loc[bad_low, ["open", "close"]].min(axis=1)
        df.loc[bad_volume, "volume"] = 0

    # Drop rows with NaN in critical columns
    pre_drop = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < pre_drop:
        logger.info(f"[clean] Dropped {pre_drop - len(df)} rows with NaN prices")

    # --- 4. Sort ---
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- 5. Add date and bar-of-day columns ---
    df["date"] = df["timestamp"].dt.date
    df["date"] = pd.to_datetime(df["date"])

    # --- 6. Outlier flagging ---
    df["is_outlier"] = False
    if len(df) > 1:
        returns = df["close"].pct_change()
        z_scores = (returns - returns.mean()) / returns.std()
        df["is_outlier"] = z_scores.abs() > spike_sigma

        n_outliers = df["is_outlier"].sum()
        if n_outliers > 0:
            logger.warning(
                f"[clean] Flagged {n_outliers} bars as outliers "
                f"(>{spike_sigma}σ return spikes)"
            )

    symbol = df["symbol"].iloc[0] if "symbol" in df.columns else "?"
    logger.info(
        f"[clean] {symbol}: {len(df)} bars, "
        f"{df['date'].nunique()} trading days, "
        f"{df['timestamp'].min()} → {df['timestamp'].max()}"
    )

    return df


def validate_session_coverage(
    df: pd.DataFrame,
    interval: str = "1h",
    min_bars_pct: float = 0.7,
) -> pd.DataFrame:
    """
    Check per-day bar counts and flag incomplete sessions.

    Args:
        df: Cleaned DataFrame with [timestamp, date, symbol].
        interval: Bar interval to determine expected bars/day.
        min_bars_pct: Minimum fraction of expected bars to keep a day.

    Returns:
        DataFrame with incomplete sessions removed.
    """
    if df.empty:
        return df

    # Expected bars per day by interval
    expected_map = {
        "5min": 78,
        "15min": 26,
        "1h": 7,  # 6.5 hours → 6 full bars + partial = ~7
        "1d": 1,
    }
    expected = expected_map.get(interval, 78)
    min_bars = int(expected * min_bars_pct)

    daily_counts = df.groupby("date").size()
    valid_days = daily_counts[daily_counts >= min_bars].index
    incomplete_days = daily_counts[daily_counts < min_bars].index

    if len(incomplete_days) > 0:
        logger.warning(
            f"[validate] Dropping {len(incomplete_days)} incomplete sessions "
            f"(< {min_bars} bars): {list(incomplete_days[:5])}..."
        )

    df = df[df["date"].isin(valid_days)].reset_index(drop=True)
    return df


def clean_and_validate(
    df: pd.DataFrame,
    interval: str = "1h",
    max_gap_fill: int = 2,
    spike_sigma: float = 8.0,
    min_bars_pct: float = 0.7,
) -> pd.DataFrame:
    """Full cleaning pipeline: clean → validate sessions."""
    df = clean_bars(df, interval, max_gap_fill, spike_sigma)
    df = validate_session_coverage(df, interval, min_bars_pct)
    return df
