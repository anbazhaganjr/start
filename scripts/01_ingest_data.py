#!/usr/bin/env python3
"""
Script 01: Ingest OHLCV data for all Ring 1 symbols.

Usage:
    python scripts/01_ingest_data.py                    # Default: yfinance, 1h bars
    python scripts/01_ingest_data.py --interval 1d      # Daily bars
    python scripts/01_ingest_data.py --providers yfinance alpaca  # Multi-source
    python scripts/01_ingest_data.py --symbols AAPL TSLA         # Specific symbols
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config
from start.data.ingest import ingest_all
from start.data.clean import clean_and_validate
from start.data.storage import save_clean, load_raw
from start.utils.logger import get_logger

logger = get_logger("01_ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest OHLCV data for START project")
    parser.add_argument(
        "--symbols", nargs="+", default=None,
        help="Symbols to fetch (default: all Ring 1 from config)"
    )
    parser.add_argument(
        "--interval", default="1h",
        choices=["5min", "15min", "1h", "1d"],
        help="Bar interval (default: 1h)"
    )
    parser.add_argument(
        "--providers", nargs="+", default=["yfinance"],
        help="Data providers in priority order (default: yfinance)"
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date YYYY-MM-DD (default: from config)"
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: from config)"
    )
    parser.add_argument(
        "--skip-clean", action="store_true",
        help="Skip cleaning step (save raw only)"
    )

    args = parser.parse_args()
    config = get_config()

    logger.info("=" * 60)
    logger.info("START Data Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"Interval:  {args.interval}")
    logger.info(f"Providers: {args.providers}")
    logger.info(f"Symbols:   {args.symbols or config['symbols']}")
    logger.info(f"Start:     {args.start or config['data']['start_date']}")
    logger.info(f"End:       {args.end or config['data']['end_date']}")
    logger.info("=" * 60)

    # Step 1: Ingest raw data
    results = ingest_all(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        interval=args.interval,
        providers=args.providers,
        save=True,
    )

    if not results:
        logger.error("No data ingested. Check provider credentials and network.")
        sys.exit(1)

    # Step 2: Clean and validate
    if not args.skip_clean:
        logger.info("")
        logger.info("Cleaning and validating...")
        for symbol, df in results.items():
            cleaned = clean_and_validate(df, interval=args.interval)
            if not cleaned.empty:
                save_clean(cleaned, symbol, args.interval)
            else:
                logger.warning(f"No valid data after cleaning for {symbol}")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    for sym, df in sorted(results.items()):
        days = df["timestamp"].dt.date.nunique() if not df.empty else 0
        logger.info(f"  {sym:6s}: {len(df):>7,} bars, {days:>4} trading days")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
