"""
Live signal generation from current market data.

Computes trading signals for each strategy using the latest feature data.
Designed to work on Streamlit Cloud (no heavy model imports required).
"""

import numpy as np
import pandas as pd
from typing import Optional

from start.utils.logger import get_logger

logger = get_logger(__name__)


def get_baseline_signals(df: pd.DataFrame) -> dict:
    """
    Compute current trading signals from baseline strategies.

    Uses the last row of the feature DataFrame to determine
    the current state of each rule-based strategy.

    Args:
        df: Feature DataFrame (at least 50+ rows for SMA computation).

    Returns:
        Dict mapping strategy name → {signal, confidence, reason}.
        signal: 1 = BUY/LONG, 0 = HOLD/FLAT, -1 = SELL (not used currently)
    """
    if df.empty or len(df) < 2:
        return {}

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    results = {}

    # 1. Buy & Hold — always long
    results["Buy & Hold"] = {
        "signal": 1,
        "label": "BUY",
        "confidence": 1.0,
        "reason": "Always invested (benchmark strategy)",
    }

    # 2. MA Crossover
    sma_20 = latest.get("sma_20", None)
    sma_50 = latest.get("sma_50", None)
    if sma_20 is not None and sma_50 is not None and not np.isnan(sma_20) and not np.isnan(sma_50):
        signal = 1 if sma_20 > sma_50 else 0
        spread = (sma_20 - sma_50) / sma_50 if sma_50 != 0 else 0
        # Check for crossover event (transition)
        prev_sma20 = prev.get("sma_20", sma_20)
        prev_sma50 = prev.get("sma_50", sma_50)
        crossed = False
        if not np.isnan(prev_sma20) and not np.isnan(prev_sma50):
            crossed = (prev_sma20 <= prev_sma50) and (sma_20 > sma_50)  # bullish cross
            crossed_bear = (prev_sma20 >= prev_sma50) and (sma_20 < sma_50)

        if crossed:
            reason = f"Bullish crossover! SMA20 ({sma_20:.2f}) just crossed above SMA50 ({sma_50:.2f})"
        elif signal == 1:
            reason = f"Uptrend: SMA20 ({sma_20:.2f}) > SMA50 ({sma_50:.2f}), spread {spread:+.2%}"
        else:
            reason = f"Downtrend: SMA20 ({sma_20:.2f}) < SMA50 ({sma_50:.2f}), spread {spread:+.2%}"

        results["MA Crossover"] = {
            "signal": signal,
            "label": "BUY" if signal == 1 else "SELL",
            "confidence": min(abs(spread) * 10, 1.0),  # Stronger spread → higher confidence
            "reason": reason,
        }

    # 3. RSI Mean Reversion
    rsi = latest.get("rsi_14", None)
    if rsi is not None and not np.isnan(rsi):
        if rsi < 30:
            signal, label = 1, "BUY"
            reason = f"Oversold: RSI = {rsi:.1f} (below 30 → expect bounce)"
            confidence = (30 - rsi) / 30  # Lower RSI → higher confidence
        elif rsi > 70:
            signal, label = 0, "SELL"
            reason = f"Overbought: RSI = {rsi:.1f} (above 70 → expect pullback)"
            confidence = (rsi - 70) / 30
        else:
            signal, label = -1, "HOLD"
            reason = f"Neutral zone: RSI = {rsi:.1f} (between 30-70)"
            confidence = 0.3

        results["RSI"] = {
            "signal": signal,
            "label": label,
            "confidence": min(confidence, 1.0),
            "reason": reason,
        }

    # 4. MACD
    macd = latest.get("macd", None)
    macd_signal = latest.get("macd_signal", None)
    macd_hist = latest.get("macd_hist", None)
    if macd is not None and macd_signal is not None and not np.isnan(macd) and not np.isnan(macd_signal):
        signal = 1 if macd > macd_signal else 0
        prev_macd = prev.get("macd", macd)
        prev_signal = prev.get("macd_signal", macd_signal)
        crossed = (not np.isnan(prev_macd) and not np.isnan(prev_signal)
                   and prev_macd <= prev_signal and macd > macd_signal)

        hist_val = macd_hist if macd_hist is not None and not np.isnan(macd_hist) else 0

        if crossed:
            reason = f"Bullish MACD crossover! Histogram: {hist_val:+.4f}"
        elif signal == 1:
            reason = f"MACD bullish: histogram = {hist_val:+.4f}"
        else:
            reason = f"MACD bearish: histogram = {hist_val:+.4f}"

        results["MACD"] = {
            "signal": signal,
            "label": "BUY" if signal == 1 else "SELL",
            "confidence": min(abs(hist_val) * 50, 1.0),
            "reason": reason,
        }

    # 5. Bollinger Band position
    bb_pct = latest.get("bb_pct", None)
    close = latest.get("close", None)
    if bb_pct is not None and not np.isnan(bb_pct) and close is not None:
        if bb_pct < 0.05:
            signal, label = 1, "BUY"
            reason = f"Price near lower Bollinger Band (BB%: {bb_pct:.2%}) — potential bounce"
            confidence = max(0.5, 1.0 - bb_pct * 10)
        elif bb_pct > 0.95:
            signal, label = 0, "SELL"
            reason = f"Price near upper Bollinger Band (BB%: {bb_pct:.2%}) — potential pullback"
            confidence = max(0.5, bb_pct)
        else:
            signal, label = -1, "HOLD"
            reason = f"Price within Bollinger Bands (BB%: {bb_pct:.2%})"
            confidence = 0.3

        results["Bollinger"] = {
            "signal": signal,
            "label": label,
            "confidence": min(confidence, 1.0),
            "reason": reason,
        }

    return results


def get_ml_signals(df: pd.DataFrame, feature_cols: Optional[list] = None) -> dict:
    """
    Train lightweight ML models on available data and predict latest signal.

    Uses walk-forward: trains on all data except last bar, predicts last bar.
    Only works locally (needs sklearn). Gracefully returns empty on Cloud.

    Args:
        df: Feature DataFrame with target column.
        feature_cols: Feature columns to use.

    Returns:
        Dict mapping model name → {signal, confidence, reason}.
    """
    try:
        from start.models.classical import LogisticModel, RidgeModel, RandomForestModel
        from start.features.builder import get_feature_columns
    except ImportError:
        logger.warning("[live] sklearn not available, skipping ML signals")
        return {}

    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    if "target" not in df.columns or len(df) < 100:
        return {}

    # Train on everything except last bar, predict last bar
    train = df.iloc[:-1].dropna(subset=feature_cols + ["target"])
    latest = df.iloc[-1:]

    if len(train) < 50:
        return {}

    X_train = train[feature_cols]
    y_train = train["target"]
    X_latest = latest[feature_cols]

    results = {}
    models = [
        ("Logistic Regression", LogisticModel()),
        ("Ridge Regression", RidgeModel()),
        ("Random Forest", RandomForestModel(n_estimators=50, max_depth=8)),
    ]

    for name, model in models:
        try:
            model.fit(X_train, y_train)
            pred = int(model.predict(X_latest)[0])
            prob = float(model.predict_proba(X_latest)[0])

            results[name] = {
                "signal": pred,
                "label": "BUY" if pred == 1 else "SELL",
                "confidence": prob if pred == 1 else 1 - prob,
                "reason": f"ML prediction: {prob:.1%} probability of price increase",
            }
        except Exception as e:
            logger.warning(f"[live] {name} failed: {e}")

    return results


def get_signal_consensus(signals: dict) -> dict:
    """
    Compute consensus across all strategy signals.

    Returns:
        Dict with: overall_signal, overall_label, agreement_pct,
        n_buy, n_sell, n_hold, strategies.
    """
    if not signals:
        return {
            "overall_signal": 0,
            "overall_label": "NO DATA",
            "agreement_pct": 0.0,
            "n_buy": 0, "n_sell": 0, "n_hold": 0,
            "strategies": {},
        }

    n_buy = sum(1 for s in signals.values() if s["signal"] == 1)
    n_sell = sum(1 for s in signals.values() if s["signal"] == 0)
    n_hold = sum(1 for s in signals.values() if s["signal"] == -1)
    total = len(signals)

    # Weighted vote (confidence-weighted)
    weighted_buy = sum(s["confidence"] for s in signals.values() if s["signal"] == 1)
    weighted_sell = sum(s["confidence"] for s in signals.values() if s["signal"] == 0)

    if weighted_buy > weighted_sell:
        overall = 1
        label = "BUY"
        agreement = n_buy / total
    elif weighted_sell > weighted_buy:
        overall = 0
        label = "SELL"
        agreement = n_sell / total
    else:
        overall = -1
        label = "HOLD"
        agreement = n_hold / total if n_hold > 0 else 0.5

    return {
        "overall_signal": overall,
        "overall_label": label,
        "agreement_pct": agreement,
        "n_buy": n_buy,
        "n_sell": n_sell,
        "n_hold": n_hold,
        "total": total,
        "strategies": signals,
    }


def get_all_signals(df: pd.DataFrame, include_ml: bool = True) -> dict:
    """
    Run all signal generators and return consensus.

    Args:
        df: Feature DataFrame.
        include_ml: Whether to include ML model signals (slower).

    Returns:
        Consensus dict from get_signal_consensus().
    """
    all_signals = {}

    # Always compute baseline signals
    baselines = get_baseline_signals(df)
    all_signals.update(baselines)

    # Optionally compute ML signals
    if include_ml:
        ml = get_ml_signals(df)
        all_signals.update(ml)

    return get_signal_consensus(all_signals)
