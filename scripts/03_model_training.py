#!/usr/bin/env python3
"""
Script 03: Train all ML models via walk-forward validation and backtest.

Usage:
    python scripts/03_train_models.py                    # All symbols, all models
    python scripts/03_train_models.py --symbols AAPL     # Single symbol
    python scripts/03_train_models.py --models logistic random_forest  # Specific models
"""

import sys
import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, get_project_root
from start.data.storage import load_features, load_clean, save_results
from start.features.builder import get_feature_columns
from start.models.baselines import buy_and_hold, ma_crossover
from start.models.training import train_all_models
from start.models.classical import evaluate_predictions
from start.backtest.engine import backtest_signals
from start.backtest.metrics import compute_metrics, format_metrics, compare_strategies
from start.backtest.regime import classify_regime, metrics_by_regime
from start.utils.logger import get_logger

logger = get_logger("03_train")


def signals_from_predictions(predictions_df: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Convert walk-forward predictions to trading signals aligned with df."""
    signals = pd.Series(0, index=df.index, name="signal")

    # Map predictions back to df indices
    for _, row in predictions_df.iterrows():
        idx = int(row["idx"])
        if idx in signals.index:
            signals.loc[idx] = int(row["predicted"])

    return signals


def main():
    parser = argparse.ArgumentParser(description="Train ML models")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None,
                        help="Models to train (logistic, ridge, random_forest, lstm, cnn)")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--train-bars", type=int, default=1500)
    parser.add_argument("--test-bars", type=int, default=200)
    parser.add_argument("--mode", default="expanding", choices=["expanding", "sliding"])
    args = parser.parse_args()

    config = get_config()
    symbols = args.symbols or config["symbols"]

    logger.info("=" * 60)
    logger.info("START Model Training Pipeline")
    logger.info(f"Mode: {args.mode} | Train: {args.train_bars} | Test: {args.test_bars}")
    logger.info("=" * 60)

    # Load SPY for regime analysis
    spy_clean = load_clean("SPY", args.interval)
    spy_regime = classify_regime(spy_clean) if not spy_clean.empty else None

    all_strategy_metrics = []

    for symbol in symbols:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# {symbol}")
        logger.info(f"{'#'*60}")

        df = load_features(symbol, args.interval)
        if df.empty or len(df) < args.train_bars + args.test_bars:
            logger.warning(f"Insufficient data for {symbol}, skipping")
            continue

        feature_cols = get_feature_columns(df)

        # ---- Baselines ----
        logger.info("\n--- Baselines ---")
        baseline_results = {}

        for bname, bfunc in [("buy_hold", buy_and_hold), ("ma_crossover", ma_crossover)]:
            signals = bfunc(df)
            bt = backtest_signals(df, signals)
            metrics = compute_metrics(bt)
            metrics["symbol"] = symbol
            metrics["strategy"] = bname
            baseline_results[bname] = metrics
            all_strategy_metrics.append(metrics)
            logger.info(f"\n{bname}:")
            logger.info(format_metrics(metrics))

        # ---- ML Models ----
        logger.info("\n--- ML Models ---")

        # Build model configs based on args
        model_configs = None
        if args.models:
            from start.models.classical import LogisticModel, RidgeModel, RandomForestModel
            from start.models.lstm import LSTMModel
            from start.models.cnn import CNNModel

            available = {
                "logistic": lambda: LogisticModel(),
                "ridge": lambda: RidgeModel(),
                "random_forest": lambda: RandomForestModel(n_estimators=50, max_depth=8),
                "lstm": lambda: LSTMModel(hidden_size=32, seq_len=20, max_epochs=30, patience=5),
                "cnn": lambda: CNNModel(seq_len=20, max_epochs=30, patience=5),
            }
            model_configs = {k: v for k, v in available.items() if k in args.models}

        ml_results = train_all_models(
            df=df,
            model_configs=model_configs,
            mode=args.mode,
            train_bars=args.train_bars,
            test_bars=args.test_bars,
            step_bars=args.test_bars,
        )

        # Backtest each ML model's predictions
        for model_name, result in ml_results.items():
            preds_df = result["predictions"]
            if preds_df.empty:
                continue

            # Create signals from predictions
            signals = signals_from_predictions(preds_df, df)

            bt = backtest_signals(df, signals)
            metrics = compute_metrics(bt)
            metrics["symbol"] = symbol
            metrics["strategy"] = model_name
            metrics["accuracy"] = result["overall_metrics"].get("accuracy", 0)
            metrics["mean_fold_accuracy"] = result["overall_metrics"].get("mean_fold_accuracy", 0)
            all_strategy_metrics.append(metrics)

            logger.info(f"\n{model_name}:")
            logger.info(format_metrics(metrics))

            # Regime analysis
            if spy_regime is not None and len(spy_regime) >= len(bt["equity_curve"]):
                regime_metrics = metrics_by_regime(
                    bt["equity_curve"],
                    spy_regime.iloc[:len(bt["equity_curve"])],
                )
                if not regime_metrics.empty:
                    logger.info(f"  Regime analysis:\n{regime_metrics.to_string()}")

    # ---- Save aggregate results ----
    if all_strategy_metrics:
        results_df = pd.DataFrame(all_strategy_metrics)
        save_results(results_df, "model_comparison")

        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON TABLE")
        logger.info("=" * 60)
        comparison = compare_strategies(
            {f"{r['symbol']}_{r['strategy']}": r for r in all_strategy_metrics}
        )
        # Show key columns
        display_cols = [
            "net_pnl", "total_return", "sharpe_ratio", "sortino_ratio",
            "max_drawdown", "win_rate", "n_trades",
        ]
        display_cols = [c for c in display_cols if c in comparison.columns]
        logger.info(f"\n{comparison[display_cols].to_string()}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
