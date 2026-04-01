#!/usr/bin/env python3
"""
Script 05: Run sentiment analysis pipeline.

Usage:
    python scripts/05_run_sentiment.py                     # All symbols
    python scripts/05_run_sentiment.py --symbols AAPL      # Single symbol
    python scripts/05_run_sentiment.py --use-ollama        # Force Ollama (must be running)
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, get_project_root
from start.sentiment.ollama_client import OllamaClient
from start.sentiment.scorer import score_symbol
from start.data.storage import save_results
from start.utils.logger import get_logger

logger = get_logger("05_sentiment")


def main():
    parser = argparse.ArgumentParser(description="Run sentiment analysis")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--use-ollama", action="store_true",
                        help="Require Ollama (fail if unavailable)")
    parser.add_argument("--marketaux-token", default=None,
                        help="Marketaux API token for live news")
    args = parser.parse_args()

    config = get_config()
    symbols = args.symbols or config["symbols"]
    root = get_project_root()

    logger.info("=" * 60)
    logger.info("START Sentiment Analysis Pipeline")
    logger.info("=" * 60)

    # Initialize Ollama client
    sent_config = config.get("sentiment", {})
    client = OllamaClient(
        model=sent_config.get("model", "mistral:7b-instruct-v0.3-q4_K_M"),
        base_url=sent_config.get("ollama_url", "http://localhost:11434"),
        timeout=sent_config.get("timeout", 30),
    )

    ollama_available = client.is_available()
    if ollama_available:
        logger.info("Ollama is available — using LLM for sentiment analysis")
    else:
        if args.use_ollama:
            logger.error("Ollama not available but --use-ollama specified. Exiting.")
            sys.exit(1)
        logger.info("Ollama not available — using Financial PhraseBank fallback")

    all_scores = []

    for symbol in symbols:
        logger.info(f"\n{'#'*40}")
        logger.info(f"# {symbol}")
        logger.info(f"{'#'*40}")

        result = score_symbol(
            symbol=symbol,
            api_token=args.marketaux_token,
            client=client if ollama_available else None,
        )

        scores = result["scores"]
        all_scores.append(scores)

        # Save per-symbol headlines
        headlines_df = result["headlines"]
        if not headlines_df.empty:
            sentiment_dir = root / "data" / "sentiment"
            sentiment_dir.mkdir(parents=True, exist_ok=True)
            headlines_df.to_parquet(
                sentiment_dir / f"{symbol}_headlines.parquet",
                engine="pyarrow",
            )

    # Save aggregate scores
    if all_scores:
        scores_df = pd.DataFrame(all_scores)
        save_results(scores_df, "sentiment_scores")

        logger.info("\n" + "=" * 60)
        logger.info("SENTIMENT SUMMARY")
        logger.info("=" * 60)
        display_cols = [
            "symbol", "mean_sentiment", "weighted_sentiment",
            "positive_pct", "negative_pct", "n_headlines",
        ]
        display_cols = [c for c in display_cols if c in scores_df.columns]
        logger.info(f"\n{scores_df[display_cols].to_string()}")

    logger.info("\n" + "=" * 60)
    logger.info("SENTIMENT ANALYSIS COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
