"""
Parquet storage layer.

Handles reading and writing data at each pipeline stage:
- raw/      → ingested OHLCV bars per symbol
- parquet/  → cleaned, validated bars per symbol
- features/ → feature matrices per symbol
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import get_config, get_project_root
from start.utils.logger import get_logger

logger = get_logger(__name__)


def _strip_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Remove timezone info from datetime columns to avoid ZoneInfoNotFoundError."""
    for col in df.columns:
        if hasattr(df[col], 'dt') and df[col].dt.tz is not None:
            df[col] = df[col].dt.tz_localize(None)
    if df.index.dtype.kind == 'M' and hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def _get_dir(stage: str) -> Path:
    """Get the directory for a pipeline stage."""
    root = get_project_root()
    config = get_config()

    stage_map = {
        "raw": Path(config["data"]["raw_dir"]),
        "parquet": Path(config["data"]["parquet_dir"]),
        "features": Path(config["data"]["features_dir"]),
        "sentiment": root / "data" / "sentiment",
        "models": root / "data" / "models",
        "results": root / "data" / "results",
    }

    d = stage_map.get(stage)
    if d is None:
        raise ValueError(f"Unknown stage '{stage}'")
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Raw data (ingested OHLCV)
# ---------------------------------------------------------------------------
def save_raw(df: pd.DataFrame, symbol: str, interval: str = "1h") -> Path:
    """Save raw ingested data to Parquet."""
    out_dir = _get_dir("raw") / "ingested"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{interval}.parquet"

    # Ensure timestamp is stored properly
    df = df.copy()
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"[storage] Saved {len(df)} raw bars → {path}")
    return path


def load_raw(symbol: str, interval: str = "1h") -> pd.DataFrame:
    """Load raw ingested data from Parquet."""
    path = _get_dir("raw") / "ingested" / f"{symbol}_{interval}.parquet"
    if not path.exists():
        logger.warning(f"[storage] No raw data found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path, engine="pyarrow")
    logger.info(f"[storage] Loaded {len(df)} raw bars from {path}")
    return df


# ---------------------------------------------------------------------------
# Cleaned data (validated OHLCV in Parquet)
# ---------------------------------------------------------------------------
def save_clean(df: pd.DataFrame, symbol: str, interval: str = "1h") -> Path:
    """Save cleaned data to Parquet."""
    out_dir = _get_dir("parquet")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}_{interval}.parquet"

    df = df.copy()
    if "timestamp" in df.columns and df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"[storage] Saved {len(df)} clean bars → {path}")
    return path


def load_clean(
    symbol: str,
    interval: str = "1h",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load cleaned data with optional date filtering."""
    path = _get_dir("parquet") / f"{symbol}_{interval}.parquet"
    if not path.exists():
        logger.warning(f"[storage] No clean data found: {path}")
        return pd.DataFrame()

    df = pd.read_parquet(path, engine="pyarrow")

    # Date filtering
    if start:
        start_dt = pd.Timestamp(start, tz="US/Eastern")
        df = df[df["timestamp"] >= start_dt]
    if end:
        end_dt = pd.Timestamp(end, tz="US/Eastern") + pd.Timedelta(days=1)
        df = df[df["timestamp"] < end_dt]

    logger.info(f"[storage] Loaded {len(df)} clean bars for {symbol}")
    return df


def load_clean_all(
    symbols: Optional[list[str]] = None,
    interval: str = "1h",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame:
    """Load cleaned data for multiple symbols into a single DataFrame."""
    if symbols is None:
        from start.utils.constants import RING1_SYMBOLS
        symbols = RING1_SYMBOLS

    dfs = []
    for sym in symbols:
        df = load_clean(sym, interval, start, end)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True).sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature data
# ---------------------------------------------------------------------------
def save_features(df: pd.DataFrame, symbol: str, interval: str = "1h") -> Path:
    """Save feature matrix to Parquet."""
    out_dir = _get_dir("features")
    path = out_dir / f"{symbol}_{interval}.parquet"

    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"[storage] Saved {len(df)} feature rows → {path}")
    return path


def _aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday bars (5min / 1h) into daily OHLCV bars."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["_date"] = df["timestamp"].dt.date

    agg = df.groupby("_date").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).reset_index()

    agg["timestamp"] = pd.to_datetime(agg["_date"])
    agg = agg.drop(columns=["_date"])

    # Carry over symbol/provider if present
    if "symbol" in df.columns:
        agg["symbol"] = df["symbol"].iloc[0]
    if "provider" in df.columns:
        agg["provider"] = df["provider"].iloc[0]

    return agg


def load_features(
    symbol: str,
    interval: str = "1h",
) -> pd.DataFrame:
    """
    Load feature matrix from Parquet.

    If interval='1d' and no daily file exists, aggregates from 1h (or 5min)
    data on the fly — so daily is always available.
    """
    path = _get_dir("features") / f"{symbol}_{interval}.parquet"

    if path.exists():
        df = pd.read_parquet(path, engine="pyarrow")
        df = _strip_tz(df)
        logger.info(f"[storage] Loaded {len(df)} feature rows for {symbol} ({interval})")
        return df

    # Fallback: aggregate intraday → daily when 1d is requested
    if interval == "1d":
        for fallback in ("1h", "5min"):
            fb_path = _get_dir("features") / f"{symbol}_{fallback}.parquet"
            if fb_path.exists():
                raw = pd.read_parquet(fb_path, engine="pyarrow")
                raw = _strip_tz(raw)
                daily = _aggregate_to_daily(raw)
                # Compute technical indicators on the daily bars
                try:
                    from start.features.technical import add_technical_indicators
                    from start.features.returns import add_returns
                    daily = add_technical_indicators(daily)
                    daily = add_returns(daily)
                    daily = daily.dropna().reset_index(drop=True)
                except Exception as e:
                    logger.warning(f"[storage] Could not compute daily indicators: {e}")
                logger.info(f"[storage] Aggregated {len(raw)} {fallback} bars → {len(daily)} daily bars for {symbol}")
                return daily

    logger.warning(f"[storage] No features found: {path}")
    return pd.DataFrame()


def load_features_all(
    symbols: Optional[list[str]] = None,
    interval: str = "1h",
) -> pd.DataFrame:
    """Load features for multiple symbols into a single DataFrame."""
    if symbols is None:
        from start.utils.constants import RING1_SYMBOLS
        symbols = RING1_SYMBOLS

    dfs = []
    for sym in symbols:
        df = load_features(sym, interval)
        if not df.empty:
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True).sort_values(
        ["symbol", "timestamp"]
    ).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Generic results / artifacts
# ---------------------------------------------------------------------------
def save_results(df: pd.DataFrame, name: str) -> Path:
    """Save a results DataFrame (backtest, ablation, etc.)."""
    out_dir = _get_dir("results")
    path = out_dir / f"{name}.parquet"
    df.to_parquet(path, index=False, engine="pyarrow")
    logger.info(f"[storage] Saved results → {path}")
    return path


def load_results(name: str) -> pd.DataFrame:
    """Load a results DataFrame."""
    path = _get_dir("results") / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path, engine="pyarrow")
    return _strip_tz(df)
