"""
Ablation study runner.

Tests 5 configurations to isolate each component's contribution:
1. Technical indicators only (MA crossover baseline)
2. Technical + ML (best classical model)
3. Technical + ML + Sentiment
4. RL only
5. Full hybrid (ensemble of all signals)
"""

from typing import Optional

import numpy as np
import pandas as pd

from start.backtest.engine import backtest_signals
from start.backtest.metrics import compute_metrics
from start.models.baselines import ma_crossover, buy_and_hold
from start.utils.logger import get_logger

logger = get_logger(__name__)


ABLATION_CONFIGS = [
    "indicators_only",
    "indicators_ml",
    "indicators_ml_sentiment",
    "rl_only",
    "full_hybrid",
]


def _ensemble_signals(*signal_series) -> pd.Series:
    """Majority vote ensemble of multiple signal series."""
    stacked = pd.DataFrame({f"s{i}": s for i, s in enumerate(signal_series)})
    # Majority vote: signal=1 if majority say 1
    majority = (stacked.sum(axis=1) > len(signal_series) / 2).astype(int)
    return majority


def run_ablation(
    df: pd.DataFrame,
    ml_signals: Optional[pd.Series] = None,
    rl_signals: Optional[pd.Series] = None,
    sentiment_score: float = 0.0,
    symbol: str = "",
) -> pd.DataFrame:
    """
    Run 5-config ablation study.

    Args:
        df: Feature DataFrame with close prices and indicators.
        ml_signals: Signal series from best ML model.
        rl_signals: Signal series from RL agent.
        sentiment_score: Aggregate sentiment (-1 to 1).
        symbol: Symbol name for logging.

    Returns:
        DataFrame with metrics per ablation config.
    """
    results = []

    logger.info(f"\n{'='*60}")
    logger.info(f"ABLATION STUDY: {symbol}")
    logger.info(f"{'='*60}")

    # Config 1: Technical indicators only (MA crossover)
    logger.info("\n[1/5] Indicators only (MA crossover)")
    ta_signals = ma_crossover(df)
    bt = backtest_signals(df, ta_signals)
    m = compute_metrics(bt)
    m["config"] = "indicators_only"
    m["symbol"] = symbol
    results.append(m)
    logger.info(f"  Sharpe: {m['sharpe_ratio']:.3f}, Return: {m['total_return']:.2%}")

    # Config 2: Indicators + ML
    if ml_signals is not None:
        logger.info("\n[2/5] Indicators + ML")
        bt = backtest_signals(df, ml_signals)
        m = compute_metrics(bt)
        m["config"] = "indicators_ml"
        m["symbol"] = symbol
        results.append(m)
        logger.info(f"  Sharpe: {m['sharpe_ratio']:.3f}, Return: {m['total_return']:.2%}")
    else:
        logger.info("\n[2/5] Indicators + ML — SKIPPED (no ML signals)")

    # Config 3: Indicators + ML + Sentiment
    if ml_signals is not None:
        logger.info("\n[3/5] Indicators + ML + Sentiment")
        # Modulate ML signals with sentiment
        # If sentiment is negative, reduce long signals
        sent_adjusted = ml_signals.copy()
        if sentiment_score < -0.3:
            # Reduce exposure in negative sentiment
            sent_adjusted = (sent_adjusted * 0.5).round().astype(int)
        elif sentiment_score > 0.3:
            # Keep full signals in positive sentiment
            pass
        bt = backtest_signals(df, sent_adjusted)
        m = compute_metrics(bt)
        m["config"] = "indicators_ml_sentiment"
        m["symbol"] = symbol
        m["sentiment_score"] = sentiment_score
        results.append(m)
        logger.info(f"  Sharpe: {m['sharpe_ratio']:.3f}, Return: {m['total_return']:.2%}")
    else:
        logger.info("\n[3/5] Indicators + ML + Sentiment — SKIPPED")

    # Config 4: RL only
    if rl_signals is not None:
        logger.info("\n[4/5] RL only")
        bt = backtest_signals(df, rl_signals)
        m = compute_metrics(bt)
        m["config"] = "rl_only"
        m["symbol"] = symbol
        results.append(m)
        logger.info(f"  Sharpe: {m['sharpe_ratio']:.3f}, Return: {m['total_return']:.2%}")
    else:
        logger.info("\n[4/5] RL only — SKIPPED (no RL signals)")

    # Config 5: Full hybrid (ensemble)
    available_signals = [ta_signals]
    if ml_signals is not None:
        available_signals.append(ml_signals)
    if rl_signals is not None:
        available_signals.append(rl_signals)

    if len(available_signals) >= 2:
        logger.info("\n[5/5] Full hybrid (ensemble)")
        hybrid = _ensemble_signals(*available_signals)
        bt = backtest_signals(df, hybrid)
        m = compute_metrics(bt)
        m["config"] = "full_hybrid"
        m["symbol"] = symbol
        m["n_components"] = len(available_signals)
        results.append(m)
        logger.info(f"  Sharpe: {m['sharpe_ratio']:.3f}, Return: {m['total_return']:.2%}")
    else:
        logger.info("\n[5/5] Full hybrid — SKIPPED (need ≥2 signal sources)")

    # Add buy-and-hold benchmark
    logger.info("\n[BM] Buy-and-hold benchmark")
    bh_signals = buy_and_hold(df)
    bt = backtest_signals(df, bh_signals)
    m = compute_metrics(bt)
    m["config"] = "buy_and_hold"
    m["symbol"] = symbol
    results.append(m)
    logger.info(f"  Sharpe: {m['sharpe_ratio']:.3f}, Return: {m['total_return']:.2%}")

    return pd.DataFrame(results)


def format_ablation_results(results_df: pd.DataFrame) -> str:
    """Format ablation results as a readable comparison table."""
    display_cols = [
        "config", "total_return", "sharpe_ratio", "sortino_ratio",
        "max_drawdown", "win_rate", "n_trades", "profit_factor",
    ]
    cols = [c for c in display_cols if c in results_df.columns]
    return results_df[cols].to_string(index=False)
