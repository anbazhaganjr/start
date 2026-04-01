#!/usr/bin/env python3
"""
Script 06: Run full backtest with ablation studies.

Usage:
    python scripts/06_run_backtest.py                      # All symbols
    python scripts/06_run_backtest.py --symbols AAPL       # Single symbol
    python scripts/06_run_backtest.py --skip-rl            # Skip RL signals
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, get_project_root
from start.data.storage import load_features, load_clean, save_results, load_results
from start.features.builder import get_feature_columns
from start.models.baselines import ma_crossover
from start.models.training import train_all_models
from start.backtest.engine import backtest_signals
from start.backtest.metrics import compute_metrics
from start.backtest.ablation import run_ablation, format_ablation_results
from start.backtest.regime import classify_regime, metrics_by_regime
from start.utils.logger import get_logger

logger = get_logger("06_backtest")


def _signals_from_predictions(predictions_df: pd.DataFrame, df: pd.DataFrame) -> pd.Series:
    """Convert ML predictions to trading signals."""
    signals = pd.Series(0, index=df.index, name="signal")
    for _, row in predictions_df.iterrows():
        idx = int(row["idx"])
        if idx in signals.index:
            signals.loc[idx] = int(row["predicted"])
    return signals


def main():
    parser = argparse.ArgumentParser(description="Run ablation backtest")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--skip-rl", action="store_true",
                        help="Skip RL signals in ablation")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment in ablation")
    parser.add_argument("--train-bars", type=int, default=1500)
    parser.add_argument("--test-bars", type=int, default=200)
    args = parser.parse_args()

    config = get_config()
    symbols = args.symbols or config["symbols"]
    root = get_project_root()
    models_dir = root / "data" / "models"

    logger.info("=" * 60)
    logger.info("START Ablation Backtest Pipeline")
    logger.info("=" * 60)

    # Load SPY for regime
    spy_clean = load_clean("SPY", args.interval)
    spy_regime = classify_regime(spy_clean) if not spy_clean.empty else None

    all_ablation_results = []

    for symbol in symbols:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# {symbol}")
        logger.info(f"{'#'*60}")

        df = load_features(symbol, args.interval)
        if df.empty or len(df) < args.train_bars + args.test_bars:
            logger.warning(f"Insufficient data for {symbol}, skipping")
            continue

        feature_cols = get_feature_columns(df)

        # Get ML signals (train best classical model)
        ml_signals = None
        try:
            from start.models.classical import LogisticModel
            ml_results = train_all_models(
                df=df,
                model_configs={"logistic": lambda: LogisticModel()},
                mode="expanding",
                train_bars=args.train_bars,
                test_bars=args.test_bars,
                step_bars=args.test_bars,
            )
            if "logistic" in ml_results and not ml_results["logistic"]["predictions"].empty:
                ml_signals = _signals_from_predictions(
                    ml_results["logistic"]["predictions"], df
                )
                logger.info(f"  ML signals: {ml_signals.sum()} long bars")
        except Exception as e:
            logger.warning(f"  ML training failed: {e}")

        # Get RL signals
        rl_signals = None
        if not args.skip_rl:
            try:
                ppo_path = models_dir / f"{symbol}_ppo.zip"
                if ppo_path.exists():
                    from stable_baselines3 import PPO
                    from start.rl.agents import generate_rl_signals, _prepare_env_data

                    model = PPO.load(str(ppo_path))
                    features, prices, mean, std = _prepare_env_data(df, feature_cols)
                    rl_signals = generate_rl_signals(
                        model=model, df=df,
                        feature_mean=mean, feature_std=std,
                        feature_cols=feature_cols,
                    )
                    logger.info(f"  RL signals: {rl_signals.sum()} long bars")
                else:
                    logger.info(f"  No RL model found at {ppo_path}")
            except Exception as e:
                logger.warning(f"  RL signal generation failed: {e}")

        # Get sentiment score
        sentiment_score = 0.0
        if not args.skip_sentiment:
            try:
                sentiment_results = load_results("sentiment_scores")
                if not sentiment_results.empty:
                    sym_sent = sentiment_results[sentiment_results["symbol"] == symbol]
                    if not sym_sent.empty:
                        sentiment_score = float(sym_sent["weighted_sentiment"].iloc[0])
                        logger.info(f"  Sentiment score: {sentiment_score:.3f}")
            except Exception:
                pass

        # Run ablation
        ablation_df = run_ablation(
            df=df,
            ml_signals=ml_signals,
            rl_signals=rl_signals,
            sentiment_score=sentiment_score,
            symbol=symbol,
        )

        all_ablation_results.append(ablation_df)

        # Regime analysis for best strategy
        if spy_regime is not None and not ablation_df.empty:
            best_config = ablation_df.sort_values("sharpe_ratio", ascending=False).iloc[0]["config"]
            logger.info(f"\n  Best config: {best_config}")

    # Save all results
    if all_ablation_results:
        final_df = pd.concat(all_ablation_results, ignore_index=True)
        save_results(final_df, "ablation_results")

        logger.info("\n" + "=" * 60)
        logger.info("ABLATION RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"\n{format_ablation_results(final_df)}")

    logger.info("\n" + "=" * 60)
    logger.info("ABLATION BACKTEST COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
