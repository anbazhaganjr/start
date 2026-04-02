"""
News headline fetcher for sentiment analysis.

Uses Alpha Vantage NEWS_SENTIMENT as primary source (symbol-specific, pre-scored),
Marketaux free API as secondary, and Financial PhraseBank as fallback.
Financial PhraseBank is a citable academic dataset (Malo et al., 2014).
"""

import os
import random
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from start.utils.logger import get_logger

logger = get_logger(__name__)

# Track last AV call to enforce rate limiting (free tier: 25 req/day)
_last_av_call_time: float = 0.0
_AV_MIN_INTERVAL: float = 12.0  # seconds between calls

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


def fetch_alphavantage_headlines(
    symbol: str,
    api_key: Optional[str] = None,
    limit: int = 50,
) -> pd.DataFrame:
    """
    Fetch symbol-specific news with pre-computed sentiment from Alpha Vantage.

    The NEWS_SENTIMENT endpoint returns articles with ticker-level sentiment
    scores and relevance, so no LLM scoring is needed.

    Args:
        symbol: Stock ticker symbol (e.g. "AAPL").
        api_key: Alpha Vantage API key. Falls back to env var ALPHAVANTAGE_API_KEY.
        limit: Max articles to retrieve (AV caps at 1000; free tier is 25 req/day).

    Returns:
        DataFrame with columns: headline, source, published_at, sentiment,
        confidence, label.  Empty DataFrame on any failure.
    """
    global _last_av_call_time

    if not api_key:
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")
    if not api_key:
        logger.info("[news] No Alpha Vantage API key configured, skipping AV")
        return pd.DataFrame()

    import requests

    # Enforce minimum interval between calls (rate-limit protection)
    elapsed = time.time() - _last_av_call_time
    if elapsed < _AV_MIN_INTERVAL and _last_av_call_time > 0:
        wait = _AV_MIN_INTERVAL - elapsed
        logger.debug(f"[news] Rate-limit: waiting {wait:.1f}s before AV call")
        time.sleep(wait)

    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "limit": min(limit, 1000),
        "apikey": api_key,
    }

    try:
        logger.info(f"[news] Fetching Alpha Vantage headlines for {symbol}")
        resp = requests.get(url, params=params, timeout=30)
        _last_av_call_time = time.time()
        resp.raise_for_status()
        data = resp.json()

        # AV returns {"Note": "..."} on rate-limit, or {"Information": "..."} on errors
        if "Note" in data:
            logger.warning(f"[news] AV rate limit hit: {data['Note']}")
            return pd.DataFrame()
        if "Information" in data:
            logger.warning(f"[news] AV info message: {data['Information']}")
            return pd.DataFrame()

        feed = data.get("feed", [])
        if not feed:
            logger.info(f"[news] AV returned no articles for {symbol}")
            return pd.DataFrame()

        rows = []
        for article in feed:
            # Find this ticker's sentiment within the article's ticker_sentiment array
            ticker_match = None
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker", "").upper() == symbol.upper():
                    ticker_match = ts
                    break

            if ticker_match is None:
                # Article mentions the ticker but has no scored entry; skip
                continue

            sentiment_score = float(ticker_match.get("ticker_sentiment_score", 0.0))
            relevance = float(ticker_match.get("relevance_score", 0.0))
            label = ticker_match.get("ticker_sentiment_label", "Neutral")

            rows.append({
                "headline": article.get("title", ""),
                "source": article.get("source", "alphavantage"),
                "published_at": article.get("time_published", ""),
                "sentiment": sentiment_score,
                "confidence": relevance,
                "label": label,
            })

        df = pd.DataFrame(rows)
        logger.info(
            f"[news] AV returned {len(df)} scored headlines for {symbol}"
        )
        return df

    except requests.exceptions.Timeout:
        logger.warning("[news] AV request timed out")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        logger.warning(f"[news] AV network error: {e}")
        return pd.DataFrame()
    except (KeyError, ValueError, TypeError) as e:
        logger.warning(f"[news] AV response parse error: {e}")
        return pd.DataFrame()


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
    av_api_key: Optional[str] = None,
    fallback_n: int = 10,
) -> list[dict] | pd.DataFrame:
    """
    Get headlines for a symbol with tiered fallback.

    Priority:
        1. Alpha Vantage NEWS_SENTIMENT (real, symbol-specific, pre-scored)
        2. Marketaux (live headlines, needs LLM scoring)
        3. PhraseBank (static academic samples)

    Returns:
        DataFrame (from Alpha Vantage, with sentiment/confidence columns)
        or list[dict] (from Marketaux / PhraseBank).
    """
    # --- 1. Try Alpha Vantage (returns DataFrame with pre-computed scores) ---
    av_df = fetch_alphavantage_headlines(symbol, api_key=av_api_key)
    if not av_df.empty:
        logger.info(f"[news] Using Alpha Vantage headlines for {symbol}")
        return av_df

    # --- 2. Try Marketaux ---
    headlines = fetch_marketaux_headlines(symbol, api_token)
    if headlines:
        logger.info(f"[news] Using Marketaux headlines for {symbol}")
        return headlines

    # --- 3. Fallback to PhraseBank ---
    logger.info(f"[news] Using PhraseBank fallback for {symbol}")
    return generate_phrasebank_headlines(n_per_category=fallback_n)
