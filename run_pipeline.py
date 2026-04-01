#!/usr/bin/env python3
"""
START Pipeline Orchestrator — Single command to run everything.

Usage:
    python run_pipeline.py                    # Full pipeline
    python run_pipeline.py --skip-sentiment   # Skip Ollama-dependent step
    python run_pipeline.py --skip-rl          # Skip RL training
    python run_pipeline.py --step 3           # Start from step 3
    python run_pipeline.py --symbols AAPL     # Single symbol
    python run_pipeline.py --quick            # Fast mode (fewer timesteps)
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import get_config, get_setting
from start.utils.logger import get_logger

logger = get_logger("pipeline")

# Pipeline steps in execution order
STEPS = [
    {
        "num": 1,
        "name": "Data Ingestion",
        "script": "scripts/01_data_ingestion.py",
        "args": [],
        "skip_flag": None,
    },
    {
        "num": 2,
        "name": "Feature Engineering",
        "script": "scripts/02_feature_engineering.py",
        "args": [],
        "skip_flag": None,
    },
    {
        "num": 3,
        "name": "Model Training (Classical + Deep Learning)",
        "script": "scripts/03_model_training.py",
        "args": [],
        "skip_flag": None,
    },
    {
        "num": 4,
        "name": "Reinforcement Learning (PPO + DQN)",
        "script": "scripts/04_reinforcement_learning.py",
        "args": [],
        "skip_flag": "skip_rl",
    },
    {
        "num": 5,
        "name": "Sentiment Analysis",
        "script": "scripts/05_sentiment_analysis.py",
        "args": [],
        "skip_flag": "skip_sentiment",
    },
    {
        "num": 6,
        "name": "Ablation Backtest",
        "script": "scripts/06_ablation_backtest.py",
        "args": [],
        "skip_flag": None,
    },
]


def run_step(step: dict, extra_args: list, dry_run: bool = False) -> bool:
    """Run a single pipeline step. Returns True on success."""
    script_path = PROJECT_ROOT / step["script"]
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False

    cmd = [sys.executable, str(script_path)] + step["args"] + extra_args

    logger.info(f"\n{'='*60}")
    logger.info(f"STEP {step['num']}: {step['name']}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"{'='*60}")

    if dry_run:
        logger.info("[DRY RUN] Skipping execution")
        return True

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=False,
            text=True,
            timeout=3600,  # 1 hour max per step
        )
        elapsed = time.time() - t0

        if result.returncode == 0:
            logger.info(f"✓ Step {step['num']} completed in {elapsed:.1f}s")
            return True
        else:
            logger.error(f"✗ Step {step['num']} failed (exit code {result.returncode})")
            return False

    except subprocess.TimeoutExpired:
        logger.error(f"✗ Step {step['num']} timed out after 3600s")
        return False
    except Exception as e:
        logger.error(f"✗ Step {step['num']} error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="START Pipeline Orchestrator — Run full pipeline with one command"
    )
    parser.add_argument("--step", type=int, default=1,
                        help="Start from this step number (1-6)")
    parser.add_argument("--symbols", nargs="+", default=None,
                        help="Override symbol list")
    parser.add_argument("--skip-rl", action="store_true",
                        help="Skip RL training step")
    parser.add_argument("--skip-sentiment", action="store_true",
                        help="Skip sentiment analysis step")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer epochs, smaller windows")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--interval", default="1h",
                        help="Data interval (default: 1h)")
    args = parser.parse_args()

    config = get_config()

    logger.info("=" * 60)
    logger.info("START PIPELINE ORCHESTRATOR")
    logger.info("=" * 60)
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Starting from step: {args.step}")
    logger.info(f"Skip RL: {args.skip_rl}")
    logger.info(f"Skip Sentiment: {args.skip_sentiment}")
    logger.info(f"Quick mode: {args.quick}")
    logger.info(f"Interval: {args.interval}")

    # Build extra args to pass through
    extra_args = []
    if args.symbols:
        extra_args += ["--symbols"] + args.symbols
    extra_args += ["--interval", args.interval]

    # Quick mode overrides
    if args.quick:
        for step in STEPS:
            if step["num"] == 3:
                step["args"] = ["--models", "logistic", "random_forest"]
            elif step["num"] == 4:
                step["args"] = ["--timesteps", "10000"]

    # Run pipeline
    total_start = time.time()
    results = []

    for step in STEPS:
        if step["num"] < args.step:
            logger.info(f"Skipping step {step['num']} (before start step)")
            continue

        # Check skip flags
        skip_attr = step.get("skip_flag")
        if skip_attr and getattr(args, skip_attr.replace("-", "_"), False):
            logger.info(f"Skipping step {step['num']}: {step['name']} (--{skip_attr})")
            results.append({"step": step["num"], "name": step["name"], "status": "SKIPPED"})
            continue

        success = run_step(step, extra_args, dry_run=args.dry_run)
        status = "OK" if success else "FAILED"
        results.append({"step": step["num"], "name": step["name"], "status": status})

        if not success and not args.dry_run:
            logger.error(f"\nPipeline stopped at step {step['num']}. Fix the error and re-run with --step {step['num']}")
            break

    # Summary
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for r in results:
        icon = "✓" if r["status"] == "OK" else "⊘" if r["status"] == "SKIPPED" else "✗"
        logger.info(f"  {icon} Step {r['step']}: {r['name']} — {r['status']}")
    logger.info(f"\nTotal time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    # Suggest next step
    all_ok = all(r["status"] in ("OK", "SKIPPED") for r in results)
    if all_ok:
        logger.info("\n🎉 Pipeline complete! Launch the dashboard:")
        logger.info(f"   streamlit run start/dashboard/app.py")
    else:
        logger.info("\n⚠ Pipeline incomplete. Fix errors above and re-run.")


if __name__ == "__main__":
    main()
