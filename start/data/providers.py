"""
Data provider abstraction layer.

Each provider implements the same interface: fetch_bars(symbol, start, end, interval).
Adding a new source (Alpaca, Tradier, Polygon, etc.) = one new class, zero changes elsewhere.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from start.utils.logger import get_logger

logger = get_logger(__name__)


class DataProvider(ABC):
    """Base class for all data providers."""

    name: str = "base"

    @abstractmethod
    def fetch_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1h",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a single symbol.

        Args:
            symbol: Ticker symbol (e.g., "AAPL").
            start: Start date as "YYYY-MM-DD".
            end: End date as "YYYY-MM-DD".
            interval: Bar interval — "5min", "15min", "1h", "1d".

        Returns:
            DataFrame with columns: [timestamp, open, high, low, close, volume]
            - timestamp is timezone-aware (US/Eastern).
            - Sorted by timestamp ascending.
            - No duplicates.
        """
        ...

    def supports_interval(self, interval: str) -> bool:
        """Check if this provider supports the given interval."""
        return interval in self.supported_intervals

    @property
    @abstractmethod
    def supported_intervals(self) -> list[str]:
        """Return list of supported interval strings."""
        ...


# ---------------------------------------------------------------------------
# yfinance provider (free, no API key)
# ---------------------------------------------------------------------------
class YFinanceProvider(DataProvider):
    """
    Yahoo Finance via yfinance.

    Limits (free tier):
        - 5min:  ~60 days back
        - 15min: ~60 days back
        - 1h:    ~730 days (2 years) back
        - 1d:    unlimited
    """

    name = "yfinance"

    # Map our interval names to yfinance format
    _INTERVAL_MAP = {
        "5min": "5m",
        "15min": "15m",
        "1h": "1h",
        "1d": "1d",
    }

    @property
    def supported_intervals(self) -> list[str]:
        return ["5min", "15min", "1h", "1d"]

    def fetch_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1h",
    ) -> pd.DataFrame:
        import yfinance as yf

        yf_interval = self._INTERVAL_MAP.get(interval)
        if yf_interval is None:
            raise ValueError(
                f"yfinance does not support interval '{interval}'. "
                f"Supported: {self.supported_intervals}"
            )

        logger.info(
            f"[yfinance] Fetching {symbol} {interval} bars: {start} → {end}"
        )

        ticker = yf.Ticker(symbol)

        # For intraday intervals, yfinance has limited lookback via start/end.
        # Use period='60d' to maximize data, then filter to requested range.
        if interval in ("5min", "15min"):
            logger.info(f"[yfinance] Using period='60d' for {interval} (maximizes lookback)")
            df = ticker.history(
                period="60d",
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,
            )
        else:
            df = ticker.history(
                start=start,
                end=end,
                interval=yf_interval,
                auto_adjust=True,
                prepost=False,  # Regular hours only
            )

        if df.empty:
            logger.warning(f"[yfinance] No data returned for {symbol} {interval}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # Normalize to our standard schema
        df = df.reset_index()

        # yfinance returns "Datetime" for intraday, "Date" for daily
        time_col = "Datetime" if "Datetime" in df.columns else "Date"
        df = df.rename(
            columns={
                time_col: "timestamp",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )

        # Keep only standard columns
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()

        # Ensure timezone-aware (US/Eastern)
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")

        # Sort and deduplicate
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df = df.reset_index(drop=True)

        logger.info(
            f"[yfinance] Got {len(df)} bars for {symbol} "
            f"({df['timestamp'].min()} → {df['timestamp'].max()})"
        )

        return df


# ---------------------------------------------------------------------------
# Alpaca provider (requires API key)
# ---------------------------------------------------------------------------
class AlpacaProvider(DataProvider):
    """
    Alpaca Markets Data API v2.

    Requires ALPACA_API_KEY and ALPACA_API_SECRET in .env.
    Free tier: IEX feed, 200 req/min, full history for 5min+ bars.
    """

    name = "alpaca"

    _INTERVAL_MAP = {
        "5min": "5Min",
        "15min": "15Min",
        "1h": "1Hour",
        "1d": "1Day",
    }

    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://data.alpaca.markets"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self._headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret,
        }

    @property
    def supported_intervals(self) -> list[str]:
        return ["5min", "15min", "1h", "1d"]

    def fetch_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1h",
    ) -> pd.DataFrame:
        import requests
        import time

        tf = self._INTERVAL_MAP.get(interval)
        if tf is None:
            raise ValueError(
                f"Alpaca does not support interval '{interval}'. "
                f"Supported: {self.supported_intervals}"
            )

        logger.info(f"[alpaca] Fetching {symbol} {interval} bars: {start} → {end}")

        all_bars = []
        page_token: Optional[str] = None

        while True:
            params = {
                "timeframe": tf,
                "start": f"{start}T00:00:00Z",
                "end": f"{end}T23:59:59Z",
                "limit": 10000,
                "feed": "iex",
                "adjustment": "all",
            }
            if page_token:
                params["page_token"] = page_token

            resp = requests.get(
                f"{self.base_url}/v2/stocks/{symbol}/bars",
                headers=self._headers,
                params=params,
            )

            if resp.status_code == 429:
                logger.warning("[alpaca] Rate limited, waiting 5s...")
                time.sleep(5)
                continue

            resp.raise_for_status()
            data = resp.json()

            bars = data.get("bars", [])
            if not bars:
                break

            all_bars.extend(bars)
            page_token = data.get("next_page_token")
            if not page_token:
                break

        if not all_bars:
            logger.warning(f"[alpaca] No data returned for {symbol} {interval}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(all_bars)
        df = df.rename(
            columns={
                "t": "timestamp",
                "o": "open",
                "h": "high",
                "l": "low",
                "c": "close",
                "v": "volume",
            }
        )

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("US/Eastern")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df = df.reset_index(drop=True)

        logger.info(
            f"[alpaca] Got {len(df)} bars for {symbol} "
            f"({df['timestamp'].min()} → {df['timestamp'].max()})"
        )

        return df


# ---------------------------------------------------------------------------
# Tradier provider (requires API token)
# ---------------------------------------------------------------------------
class TradierProvider(DataProvider):
    """
    Tradier Markets API.

    Requires TRADIER_API_KEY in .env.
    Supports 5min intraday bars. Daily bars available via /v1/markets/history.
    """

    name = "tradier"

    _INTERVAL_MAP_INTRADAY = {
        "5min": "5min",
        "15min": "15min",
    }

    def __init__(self, api_token: str, sandbox: bool = False):
        self.api_token = api_token
        base = "https://sandbox.tradier.com" if sandbox else "https://api.tradier.com"
        self.base_url = base
        self._headers = {
            "Authorization": f"Bearer {api_token}",
            "Accept": "application/json",
        }

    @property
    def supported_intervals(self) -> list[str]:
        return ["5min", "15min", "1d"]

    def fetch_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "5min",
    ) -> pd.DataFrame:
        import requests
        import time

        if interval == "1d":
            return self._fetch_daily(symbol, start, end)

        tradier_interval = self._INTERVAL_MAP_INTRADAY.get(interval)
        if tradier_interval is None:
            raise ValueError(
                f"Tradier does not support interval '{interval}'. "
                f"Supported: {self.supported_intervals}"
            )

        logger.info(f"[tradier] Fetching {symbol} {interval} bars: {start} → {end}")

        # Tradier timesales supports date-range queries (max ~1 month per call)
        # Chunk into ~3-week windows to stay within API limits
        all_dfs = []
        current = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)
        chunk_days = 20  # ~3 weeks per request

        while current <= end_dt:
            chunk_end = min(current + timedelta(days=chunk_days), end_dt)
            start_str = current.strftime("%Y-%m-%d")
            end_str = chunk_end.strftime("%Y-%m-%d")

            logger.info(f"[tradier] Chunk: {start_str} → {end_str}")

            resp = requests.get(
                f"{self.base_url}/v1/markets/timesales",
                params={
                    "symbol": symbol,
                    "interval": tradier_interval,
                    "start": start_str,
                    "end": end_str,
                },
                headers=self._headers,
            )

            if resp.status_code == 429:
                logger.warning("[tradier] Rate limited, waiting 5s...")
                time.sleep(5)
                continue  # retry same chunk

            if resp.status_code == 200:
                data = resp.json()
                if "series" in data and data["series"] and "data" in data["series"]:
                    bars = data["series"]["data"]
                    day_df = pd.DataFrame(bars)
                    all_dfs.append(day_df)
                    logger.info(f"[tradier] Got {len(bars)} bars for chunk")
            else:
                logger.warning(f"[tradier] HTTP {resp.status_code} for {symbol} chunk {start_str}-{end_str}")

            current = chunk_end + timedelta(days=1)
            time.sleep(0.5)  # be polite to the API

        if not all_dfs:
            logger.warning(f"[tradier] No data returned for {symbol} {interval}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.concat(all_dfs, ignore_index=True)
        # Tradier returns 'time' (ISO string) and 'timestamp' (unix epoch)
        # Use 'time' column for parsing, drop the unix 'timestamp'
        if "time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["time"])
            df = df.drop(columns=["time"], errors="ignore")
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Strip timezone if present, then localize
        if df["timestamp"].dt.tz is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert("US/Eastern")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        # Filter to market hours only (09:30 - 16:00 ET)
        df = df[
            (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute >= 570)  # 9:30
            & (df["timestamp"].dt.hour * 60 + df["timestamp"].dt.minute <= 960)  # 16:00
        ]
        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df = df.reset_index(drop=True)

        logger.info(f"[tradier] Got {len(df)} total bars for {symbol}")
        return df

    def _fetch_daily(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        import requests

        resp = requests.get(
            f"{self.base_url}/v1/markets/history",
            params={
                "symbol": symbol,
                "interval": "daily",
                "start": start,
                "end": end,
            },
            headers=self._headers,
        )
        resp.raise_for_status()
        data = resp.json()

        if "history" not in data or not data["history"]:
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.DataFrame(data["history"]["day"])
        df = df.rename(columns={"date": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("US/Eastern")
        df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Local file provider (reads existing raw data from disk)
# ---------------------------------------------------------------------------
class LocalFileProvider(DataProvider):
    """
    Reads existing raw CSV/JSON files from data/raw/.

    Useful for incorporating previously-fetched Tradier data without needing
    a live API connection. Supports both CSV (tradier/) and JSON (tradier_json/).
    """

    name = "local"

    def __init__(self, raw_dir: str):
        self.raw_dir = raw_dir

    @property
    def supported_intervals(self) -> list[str]:
        return ["5min"]  # Existing data is 5-min bars

    def fetch_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "5min",
    ) -> pd.DataFrame:
        from pathlib import Path
        import glob
        import json

        all_dfs = []

        # Read CSV files (data/raw/tradier/)
        csv_dir = Path(self.raw_dir) / "tradier"
        if csv_dir.exists():
            pattern = str(csv_dir / f"{symbol}_*.csv")
            for fpath in sorted(glob.glob(pattern)):
                df = pd.read_csv(fpath)
                if "time" in df.columns:
                    df = df.rename(columns={"time": "timestamp"})
                all_dfs.append(df)

        # Read JSON files (data/raw/tradier_json/{SYMBOL}/)
        json_dir = Path(self.raw_dir) / "tradier_json" / symbol
        if json_dir.exists():
            for fpath in sorted(json_dir.glob(f"{symbol}_*.json")):
                with open(fpath) as f:
                    data = json.load(f)

                # Tradier JSON structure: {"series": {"data": [...]}}
                if isinstance(data, dict):
                    if "series" in data and data["series"]:
                        bars = data["series"]["data"]
                    elif "data" in data:
                        bars = data["data"]
                    else:
                        bars = [data] if "open" in data else []
                elif isinstance(data, list):
                    bars = data
                else:
                    continue

                if bars:
                    df = pd.DataFrame(bars)
                    if "time" in df.columns:
                        df = df.rename(columns={"time": "timestamp"})
                    all_dfs.append(df)

        if not all_dfs:
            logger.warning(f"[local] No files found for {symbol}")
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        df = pd.concat(all_dfs, ignore_index=True)

        # Normalize timestamp
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("US/Eastern")

        # Keep standard columns
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        df = df[cols].copy()

        # Filter date range
        start_dt = pd.Timestamp(start, tz="US/Eastern")
        end_dt = pd.Timestamp(end, tz="US/Eastern") + timedelta(days=1)
        df = df[(df["timestamp"] >= start_dt) & (df["timestamp"] < end_dt)]

        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        df = df.reset_index(drop=True)

        logger.info(f"[local] Loaded {len(df)} bars for {symbol} from disk")
        return df


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_provider(name: str, config: Optional[dict] = None) -> DataProvider:
    """
    Factory to create a data provider by name.

    Args:
        name: One of "yfinance", "alpaca", "tradier", "local".
        config: Project config dict (from config.get_config()).

    Returns:
        Initialized DataProvider instance.
    """
    if config is None:
        from config import get_config
        config = get_config()

    if name == "yfinance":
        return YFinanceProvider()

    elif name == "alpaca":
        api = config.get("api", {})
        key = api.get("alpaca_key", "")
        secret = api.get("alpaca_secret", "")
        base_url = api.get("alpaca_base_url", "https://data.alpaca.markets")
        if not key or not secret:
            raise ValueError(
                "Alpaca API key/secret not set. Add ALPACA_API_KEY and "
                "ALPACA_API_SECRET to your .env file."
            )
        return AlpacaProvider(api_key=key, api_secret=secret, base_url=base_url)

    elif name == "tradier":
        api = config.get("api", {})
        token = api.get("tradier_key", "")
        if not token:
            raise ValueError(
                "Tradier API token not set. Add TRADIER_API_KEY to your .env file."
            )
        return TradierProvider(api_token=token)

    elif name == "local":
        raw_dir = config["data"]["raw_dir"]
        return LocalFileProvider(raw_dir=raw_dir)

    else:
        raise ValueError(
            f"Unknown provider '{name}'. Choose from: yfinance, alpaca, tradier, local"
        )
