"""
Sentiment scoring pipeline.

Processes headlines through LLM, aggregates scores per bar,
and aligns with feature DataFrames.
"""

from typing import Optional

import numpy as np
import pandas as pd

from start.sentiment.ollama_client import OllamaClient
from start.sentiment.news_fetcher import get_headlines_for_symbol, generate_phrasebank_headlines
from start.utils.logger import get_logger

logger = get_logger(__name__)


def score_headlines(
    headlines: list[dict],
    client: Optional[OllamaClient] = None,
) -> pd.DataFrame:
    """
    Score a list of headlines using the LLM.

    If Ollama is not available, uses true_sentiment from PhraseBank fallback.

    Args:
        headlines: List of headline dicts.
        client: Optional OllamaClient. Created if not provided.

    Returns:
        DataFrame with columns: headline, sentiment, confidence, source.
    """
    if client is None:
        client = OllamaClient()

    use_llm = client.is_available()

    if use_llm:
        logger.info(f"[sentiment] Scoring {len(headlines)} headlines with Ollama")
    else:
        logger.info(f"[sentiment] Ollama not available, using ground truth labels")

    results = []
    for h in headlines:
        if use_llm:
            score = client.analyze_sentiment(h["headline"])
        else:
            # Use true sentiment if available (PhraseBank), else neutral
            score = {
                "sentiment": h.get("true_sentiment", 0),
                "confidence": 0.8 if "true_sentiment" in h else 0.3,
            }

        results.append({
            "headline": h["headline"],
            "sentiment": score["sentiment"],
            "confidence": score["confidence"],
            "source": h.get("source", "unknown"),
        })

    return pd.DataFrame(results)


def compute_sentiment_score(scored_df: pd.DataFrame) -> dict:
    """
    Aggregate headline scores into a single sentiment signal.

    Returns:
        Dict with aggregate metrics.
    """
    if scored_df.empty:
        return {
            "mean_sentiment": 0.0,
            "weighted_sentiment": 0.0,
            "positive_pct": 0.0,
            "negative_pct": 0.0,
            "neutral_pct": 0.0,
            "n_headlines": 0,
            "mean_confidence": 0.0,
        }

    sentiments = scored_df["sentiment"]
    confidences = scored_df["confidence"]

    # Weighted mean (by confidence)
    if confidences.sum() > 0:
        weighted = (sentiments * confidences).sum() / confidences.sum()
    else:
        weighted = sentiments.mean()

    n = len(scored_df)
    return {
        "mean_sentiment": sentiments.mean(),
        "weighted_sentiment": weighted,
        "positive_pct": (sentiments > 0).sum() / n,
        "negative_pct": (sentiments < 0).sum() / n,
        "neutral_pct": (sentiments == 0).sum() / n,
        "n_headlines": n,
        "mean_confidence": confidences.mean(),
    }


def score_symbol(
    symbol: str,
    api_token: Optional[str] = None,
    client: Optional[OllamaClient] = None,
) -> dict:
    """
    Full sentiment pipeline for a single symbol.

    Args:
        symbol: Stock ticker.
        api_token: Optional Marketaux API token.
        client: Optional OllamaClient.

    Returns:
        Dict with scored headlines DataFrame and aggregate metrics.
    """
    logger.info(f"[sentiment] Processing {symbol}")

    # Fetch headlines
    headlines = get_headlines_for_symbol(symbol, api_token)

    if not headlines:
        logger.warning(f"[sentiment] No headlines for {symbol}")
        return {
            "symbol": symbol,
            "headlines": pd.DataFrame(),
            "scores": compute_sentiment_score(pd.DataFrame()),
        }

    # Score headlines
    scored = score_headlines(headlines, client)

    # Aggregate
    scores = compute_sentiment_score(scored)
    scores["symbol"] = symbol

    logger.info(
        f"[sentiment] {symbol}: "
        f"mean={scores['mean_sentiment']:.3f}, "
        f"weighted={scores['weighted_sentiment']:.3f}, "
        f"pos={scores['positive_pct']:.1%}, neg={scores['negative_pct']:.1%}, "
        f"n={scores['n_headlines']}"
    )

    return {
        "symbol": symbol,
        "headlines": scored,
        "scores": scores,
    }


def add_sentiment_to_features(
    features_df: pd.DataFrame,
    sentiment_score: float,
) -> pd.DataFrame:
    """
    Add a constant sentiment score to a feature DataFrame.

    For intraday data, sentiment is typically daily or slower —
    so we broadcast the current aggregate score across all bars.

    Args:
        features_df: Feature DataFrame.
        sentiment_score: Aggregate weighted sentiment (-1 to 1).

    Returns:
        DataFrame with 'sentiment' column added.
    """
    df = features_df.copy()
    df["sentiment"] = sentiment_score
    return df
