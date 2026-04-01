"""
News headline fetcher for sentiment analysis.

Uses Marketaux free API as primary source with Financial PhraseBank as fallback.
Financial PhraseBank is a citable academic dataset (Malo et al., 2014).
"""

import json
import random
from pathlib import Path
from typing import Optional

import pandas as pd

from start.utils.logger import get_logger

logger = get_logger(__name__)

# Financial PhraseBank sample headlines for fallback
# Source: Malo et al., "Good debt or bad debt: Detecting semantic orientations in economic texts" (2014)
PHRASEBANK_SAMPLES = {
    "positive": [
        "The company reported record quarterly earnings, beating analyst expectations.",
        "Revenue increased 15% year-over-year driven by strong demand.",
        "The stock surged after announcing a major partnership deal.",
        "Management raised full-year guidance citing robust market conditions.",
        "Net income grew significantly as cost reduction measures took effect.",
        "The company expanded into new markets with promising early results.",
        "Strong consumer spending boosted retail sector performance.",
        "The firm secured a $2 billion contract, its largest ever.",
        "Analysts upgraded the stock to buy following impressive quarterly results.",
        "The company announced a share buyback program worth $500 million.",
    ],
    "negative": [
        "The company issued a profit warning citing supply chain disruptions.",
        "Revenue declined 10% as demand softened across key markets.",
        "The stock dropped after missing earnings estimates for the third quarter.",
        "Management cut guidance due to rising costs and weakening demand.",
        "Net losses widened as restructuring charges weighed on results.",
        "The company announced layoffs affecting 5,000 employees.",
        "Weak consumer confidence led to lower than expected sales.",
        "The firm lost a major contract to a competitor.",
        "Analysts downgraded the stock citing deteriorating fundamentals.",
        "The company suspended its dividend to preserve cash.",
    ],
    "neutral": [
        "The company reported results in line with analyst expectations.",
        "Revenue remained flat compared to the previous quarter.",
        "The board appointed a new chief financial officer.",
        "The company completed its previously announced acquisition.",
        "Trading volume was average with no significant price movement.",
        "The firm maintained its current dividend payout ratio.",
        "Quarterly results showed mixed performance across business segments.",
        "The company filed its annual report with the SEC.",
        "Industry analysts maintained their neutral outlook on the sector.",
        "The company announced a routine change to its board of directors.",
    ],
}


def fetch_marketaux_headlines(
    symbol: str,
    api_token: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch headlines from Marketaux API.

    Args:
        symbol: Stock ticker symbol.
        api_token: Marketaux API token (free tier).
        limit: Max headlines to fetch.

    Returns:
        List of dicts with 'headline', 'published_at', 'source'.
    """
    if not api_token:
        logger.info("[news] No Marketaux API token, using fallback dataset")
        return []

    import requests

    url = "https://api.marketaux.com/v1/news/all"
    params = {
        "symbols": symbol,
        "filter_entities": True,
        "language": "en",
        "api_token": api_token,
        "limit": min(limit, 50),  # Free tier max
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        headlines = []
        for article in data.get("data", []):
            headlines.append({
                "headline": article.get("title", ""),
                "published_at": article.get("published_at", ""),
                "source": article.get("source", "marketaux"),
            })

        logger.info(f"[news] Fetched {len(headlines)} headlines for {symbol}")
        return headlines

    except Exception as e:
        logger.warning(f"[news] Marketaux fetch failed: {e}")
        return []


def generate_phrasebank_headlines(
    n_per_category: int = 10,
    seed: int = 42,
) -> list[dict]:
    """
    Generate sample headlines from Financial PhraseBank dataset.

    This provides a reproducible fallback when live news APIs are unavailable.

    Args:
        n_per_category: Headlines per sentiment category.
        seed: Random seed for reproducibility.

    Returns:
        List of dicts with 'headline', 'true_sentiment', 'source'.
    """
    rng = random.Random(seed)
    headlines = []

    for sentiment, samples in PHRASEBANK_SAMPLES.items():
        selected = samples[:n_per_category]
        for headline in selected:
            sentiment_val = {"positive": 1, "negative": -1, "neutral": 0}[sentiment]
            headlines.append({
                "headline": headline,
                "true_sentiment": sentiment_val,
                "source": "phrasebank",
            })

    rng.shuffle(headlines)
    return headlines


def get_headlines_for_symbol(
    symbol: str,
    api_token: Optional[str] = None,
    fallback_n: int = 10,
) -> list[dict]:
    """
    Get headlines for a symbol, with automatic fallback.

    Tries Marketaux first, falls back to PhraseBank samples.
    """
    # Try live API
    headlines = fetch_marketaux_headlines(symbol, api_token)
    if headlines:
        return headlines

    # Fallback to PhraseBank
    logger.info(f"[news] Using PhraseBank fallback for {symbol}")
    return generate_phrasebank_headlines(n_per_category=fallback_n)
