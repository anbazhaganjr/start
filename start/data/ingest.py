"""
Data ingestion orchestrator.

Fetches OHLCV bars for all configured symbols using a priority chain of
providers (yfinance → alpaca → tradier → local). Results are merged and
deduplicated, then passed to the storage layer.
"""

from typing import Optional

import pandas as pd
from tqdm import tqdm

from config import get_config
from start.data.providers import DataProvider, get_provider
from start.data.storage import save_raw, load_raw
from start.utils.logger import get_logger

logger = get_logger(__name__)


def ingest_symbol(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1h",
    providers: Optional[list[str]] = None,
    config: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Fetch bars for a single symbol, trying providers in priority order.

    Merges results from all successful providers, deduplicates on timestamp,
    and returns a single clean DataFrame.

    Args:
        symbol: Ticker (e.g., "AAPL").
        start: Start date "YYYY-MM-DD".
        end: End date "YYYY-MM-DD".
        interval: Bar interval.
        providers: Provider names in priority order. Defaults to ["yfinance"].
        config: Project config dict.

    Returns:
        DataFrame with columns [timestamp, open, high, low, close, volume, symbol, provider].
    """
    if config is None:
        config = get_config()

    if providers is None:
        providers = ["yfinance"]

    all_dfs = []

    for pname in providers:
        try:
            provider = get_provider(pname, config)

            if not provider.supports_interval(interval):
                logger.warning(
                    f"[{pname}] Does not support {interval} for {symbol}, skipping"
                )
                continue

            df = provider.fetch_bars(symbol, start, end, interval)

            if not df.empty:
                df["symbol"] = symbol
                df["provider"] = pname
                all_dfs.append(df)

        except Exception as e:
            logger.error(f"[{pname}] Failed for {symbol}: {e}")
            continue

    if not all_dfs:
        logger.warning(f"No data from any provider for {symbol} {interval}")
        return pd.DataFrame()

    # Merge: use first provider's data as primary, fill gaps from others
    combined = pd.concat(all_dfs, ignore_index=True)
    combined = combined.sort_values(["timestamp", "provider"])

    # Keep first occurrence per timestamp (respects provider priority order)
    combined = combined.drop_duplicates(subset=["timestamp"], keep="first")
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        f"[ingest] {symbol}: {len(combined)} bars total "
        f"({combined['timestamp'].min()} → {combined['timestamp'].max()})"
    )

    return combined


def ingest_all(
    symbols: Optional[list[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    interval: str = "1h",
    providers: Optional[list[str]] = None,
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch bars for all symbols and optionally save to disk.

    Args:
        symbols: List of tickers. Defaults to config["symbols"].
        start: Start date. Defaults to config["data"]["start_date"].
        end: End date. Defaults to config["data"]["end_date"].
        interval: Bar interval.
        providers: Provider priority list.
        save: If True, save each symbol's data to raw Parquet.

    Returns:
        Dict mapping symbol → DataFrame.
    """
    config = get_config()

    if symbols is None:
        symbols = config["symbols"]
    if start is None:
        start = config["data"]["start_date"]
    if end is None:
        end = config["data"]["end_date"]
    if providers is None:
        providers = ["yfinance"]

    results = {}

    for symbol in tqdm(symbols, desc="Ingesting"):
        df = ingest_symbol(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval,
            providers=providers,
            config=config,
        )

        if not df.empty:
            results[symbol] = df
            if save:
                save_raw(df, symbol, interval)
        else:
            logger.warning(f"No data for {symbol}, skipping save")

    # Summary
    total_bars = sum(len(df) for df in results.values())
    logger.info(
        f"[ingest] Complete: {len(results)}/{len(symbols)} symbols, "
        f"{total_bars:,} total bars"
    )

    return results
