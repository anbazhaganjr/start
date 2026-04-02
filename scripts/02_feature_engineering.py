#!/usr/bin/env python3
"""
Script 02: Build feature matrices from cleaned OHLCV data.

Usage:
    python scripts/02_build_features.py                  # All symbols, 1h
    python scripts/02_build_features.py --interval 1d    # Daily features
    python scripts/02_build_features.py --symbols AAPL TSLA
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config
from start.data.storage import load_clean, save_features
from start.features.builder import build_features, get_feature_columns, generate_summary
from start.utils.logger import get_logger

logger = get_logger("02_features")


def main():
    parser = argparse.ArgumentParser(description="Build feature matrices")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--interval", default="1h", choices=["5min", "15min", "1h", "1d"])
    parser.add_argument("--no-target", action="store_true", help="Skip target creation")
    parser.add_argument("--drop-correlated", action="store_true", help="Drop highly correlated features")
    args = parser.parse_args()

    config = get_config()
    symbols = args.symbols or config["symbols"]

    logger.info("=" * 60)
    logger.info("START Feature Engineering Pipeline")
    logger.info("=" * 60)

    for symbol in symbols:
        logger.info(f"\n--- {symbol} ---")

        df = load_clean(symbol, args.interval)
        if df.empty:
            logger.warning(f"No clean data for {symbol}, skipping")
            continue

        features_df = build_features(
            df,
            add_target=not args.no_target,
            drop_correlated=args.drop_correlated,
        )

        if not features_df.empty:
            save_features(features_df, symbol, args.interval)

            # Print summary
            feat_cols = get_feature_columns(features_df)
            logger.info(
                f"{symbol}: {len(features_df)} rows, "
                f"{len(feat_cols)} features, "
                f"{features_df['timestamp'].min()} → {features_df['timestamp'].max()}"
            )
            if "target" in features_df.columns:
                up_pct = features_df["target"].mean()
                logger.info(f"  Target balance: {up_pct:.1%} up / {1-up_pct:.1%} down")

    logger.info("\n" + "=" * 60)
    logger.info("FEATURE BUILD COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
