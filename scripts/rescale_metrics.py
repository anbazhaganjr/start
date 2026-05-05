"""
Rescale Sharpe/Sortino/volatility/calmar metrics from the legacy 5-min
annualization factor (252 * 78 = 19656) to the correct hourly factor
(252 * 7 = 1764). This is mathematically identical to recomputing metrics
from the underlying equity curves with the corrected ann_factor.

Scaling rules:
  - sqrt(ann_factor) metrics: sharpe_ratio, sortino_ratio, volatility
      → multiply by sqrt(1764 / 19656) ≈ 0.2996
  - linear ann_factor metrics: annualized_return
      → multiply by (1764 / 19656) ≈ 0.0897
  - calmar_ratio = annualized_return / max_drawdown → recomputed
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

OLD_FACTOR = 252 * 78    # = 19656 (5-min)
NEW_FACTOR = 252 * 7     # = 1764  (1-hour)

SQRT_SCALE = np.sqrt(NEW_FACTOR / OLD_FACTOR)   # ≈ 0.2996
LIN_SCALE = NEW_FACTOR / OLD_FACTOR             # ≈ 0.0897

print(f"Annualization correction:")
print(f"  Old: {OLD_FACTOR} (5-min bars)")
print(f"  New: {NEW_FACTOR} (hourly bars)")
print(f"  sqrt-scale (Sharpe/Sortino/Vol): {SQRT_SCALE:.4f}")
print(f"  linear-scale (Annual Return):   {LIN_SCALE:.4f}")
print()

RESULTS = Path("data/results")
SQRT_COLS = ["sharpe_ratio", "sortino_ratio", "volatility"]
LIN_COLS = ["annualized_return"]

for fname in ["model_comparison.parquet", "ablation_results.parquet", "rl_comparison.parquet"]:
    p = RESULTS / fname
    df = pd.read_parquet(p)
    print(f"--- {fname} ---")
    print(f"  rows: {len(df)}, cols: {len(df.columns)}")

    for col in SQRT_COLS:
        if col in df.columns:
            df[col] = df[col] * SQRT_SCALE
            print(f"  rescaled {col} by {SQRT_SCALE:.4f}")

    for col in LIN_COLS:
        if col in df.columns:
            df[col] = df[col] * LIN_SCALE
            print(f"  rescaled {col} by {LIN_SCALE:.4f}")

    if "annualized_return" in df.columns and "max_drawdown" in df.columns:
        df["calmar_ratio"] = np.where(
            df["max_drawdown"] > 0,
            df["annualized_return"] / df["max_drawdown"],
            0.0,
        )
        print(f"  recomputed calmar_ratio from corrected annualized_return")

    df.to_parquet(p, index=False)
    print(f"  saved → {p}")
    print()

# Sentiment scores parquet has no Sharpe-related metrics; leave as-is.
print("Sentiment scores parquet has no annualized metrics — unchanged.")
print()
print("Done.")
