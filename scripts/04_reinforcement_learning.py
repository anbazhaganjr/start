#!/usr/bin/env python3
"""
Script 04: Train RL agents (PPO + DQN) via stable-baselines3.

Usage:
    python scripts/04_train_rl.py                         # All symbols, both agents
    python scripts/04_train_rl.py --symbols AAPL           # Single symbol
    python scripts/04_train_rl.py --agents ppo             # PPO only
    python scripts/04_train_rl.py --timesteps 100000       # More training
"""

import sys
import argparse
import time
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_config, get_project_root
from start.data.storage import load_features, save_results
from start.features.builder import get_feature_columns
from start.rl.agents import train_ppo, train_dqn, generate_rl_signals
from start.backtest.engine import backtest_signals
from start.backtest.metrics import compute_metrics, format_metrics
from start.utils.logger import get_logger

logger = get_logger("04_rl")


def main():
    parser = argparse.ArgumentParser(description="Train RL agents")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--agents", nargs="+", default=["ppo", "dqn"],
                        choices=["ppo", "dqn"])
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--window-size", type=int, default=20)
    args = parser.parse_args()

    config = get_config()
    symbols = args.symbols or config["symbols"]
    root = get_project_root()
    models_dir = root / "data" / "models"

    logger.info("=" * 60)
    logger.info("START RL Training Pipeline")
    logger.info(f"Agents: {args.agents} | Timesteps: {args.timesteps}")
    logger.info("=" * 60)

    all_results = []

    for symbol in symbols:
        logger.info(f"\n{'#'*60}")
        logger.info(f"# {symbol}")
        logger.info(f"{'#'*60}")

        df = load_features(symbol, args.interval)
        if df.empty or len(df) < 500:
            logger.warning(f"Insufficient data for {symbol}, skipping")
            continue

        feature_cols = get_feature_columns(df)

        for agent_name in args.agents:
            logger.info(f"\n--- Training {agent_name.upper()} for {symbol} ---")
            t0 = time.time()

            save_path = str(models_dir / f"{symbol}_{agent_name}")

            try:
                if agent_name == "ppo":
                    result = train_ppo(
                        df=df,
                        total_timesteps=args.timesteps,
                        window_size=args.window_size,
                        feature_cols=feature_cols,
                        save_path=save_path,
                    )
                else:
                    result = train_dqn(
                        df=df,
                        total_timesteps=args.timesteps,
                        window_size=args.window_size,
                        feature_cols=feature_cols,
                        save_path=save_path,
                    )

                elapsed = time.time() - t0
                eval_m = result["eval_metrics"]

                # Generate signals and backtest
                signals = generate_rl_signals(
                    model=result["model"],
                    df=df,
                    feature_mean=result["feature_mean"],
                    feature_std=result["feature_std"],
                    feature_cols=feature_cols,
                    window_size=args.window_size,
                )

                bt = backtest_signals(df, signals)
                metrics = compute_metrics(bt)
                metrics["symbol"] = symbol
                metrics["strategy"] = f"rl_{agent_name}"
                metrics["train_time"] = elapsed
                metrics["eval_return"] = eval_m["total_return"]
                metrics["eval_equity"] = eval_m["final_equity"]
                all_results.append(metrics)

                logger.info(f"\n{agent_name.upper()} Results:")
                logger.info(format_metrics(metrics))
                logger.info(f"  Training time: {elapsed:.1f}s")

            except Exception as e:
                logger.error(f"Failed to train {agent_name} for {symbol}: {e}")
                import traceback
                traceback.print_exc()

    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        save_results(results_df, "rl_comparison")

        logger.info("\n" + "=" * 60)
        logger.info("RL COMPARISON TABLE")
        logger.info("=" * 60)
        display_cols = [
            "symbol", "strategy", "net_pnl", "total_return",
            "sharpe_ratio", "max_drawdown", "n_trades",
        ]
        display_cols = [c for c in display_cols if c in results_df.columns]
        logger.info(f"\n{results_df[display_cols].to_string()}")

    logger.info("\n" + "=" * 60)
    logger.info("RL TRAINING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
