"""
Cost-aware backtesting engine.

Simulates trade execution with slippage and commission costs.
Produces equity curves, trade logs, and performance metrics.
"""

import numpy as np
import pandas as pd

from start.utils.constants import SLIPPAGE_PCT, COMMISSION_PER_SHARE
from start.utils.logger import get_logger

logger = get_logger(__name__)


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 100_000.0,
    slippage_pct: float = SLIPPAGE_PCT,
    commission_per_share: float = COMMISSION_PER_SHARE,
    shares_per_trade: int = 100,
) -> dict:
    """
    Backtest a signal series against price data with transaction costs.

    Args:
        df: DataFrame with [timestamp, close] columns.
        signals: Series of signals (1=long, 0=flat, -1=short).
        initial_capital: Starting capital in USD.
        slippage_pct: Slippage as fraction of price per trade.
        commission_per_share: Commission per share per trade.
        shares_per_trade: Position size in shares.

    Returns:
        Dict with:
            - equity_curve: DataFrame [timestamp, equity, position, drawdown]
            - trades: DataFrame [entry_time, exit_time, entry_price, exit_price, pnl, cost]
            - metrics: Dict of performance metrics
    """
    n = len(df)
    if n != len(signals):
        raise ValueError(f"Length mismatch: df={n}, signals={len(signals)}")

    close = df["close"].values.astype(float)
    timestamps = df["timestamp"].values

    # Track state
    equity = np.full(n, initial_capital, dtype=float)
    positions = np.zeros(n, dtype=int)  # 0=flat, 1=long
    drawdowns = np.zeros(n, dtype=float)

    # Trade log
    trades = []
    current_position = 0
    entry_price = 0.0
    entry_time = None
    total_costs = 0.0

    peak_equity = initial_capital
    cash = initial_capital
    shares_held = 0

    for i in range(n):
        signal = int(signals.iloc[i]) if hasattr(signals, "iloc") else int(signals[i])
        price = close[i]

        # Position change?
        if signal != current_position:
            cost = 0.0

            # Close existing position
            if current_position == 1 and shares_held > 0:
                sell_price = price * (1 - slippage_pct)
                cost += commission_per_share * shares_held
                proceeds = sell_price * shares_held - cost
                pnl = proceeds - (entry_price * shares_held)

                trades.append({
                    "entry_time": entry_time,
                    "exit_time": timestamps[i],
                    "entry_price": entry_price,
                    "exit_price": sell_price,
                    "shares": shares_held,
                    "pnl": pnl,
                    "cost": cost,
                    "return_pct": pnl / (entry_price * shares_held) if entry_price > 0 else 0,
                })

                cash += proceeds
                shares_held = 0
                total_costs += cost

            # Open new position
            if signal == 1:
                buy_price = price * (1 + slippage_pct)
                cost = commission_per_share * shares_per_trade
                total_cost = buy_price * shares_per_trade + cost

                if cash >= total_cost:
                    cash -= total_cost
                    shares_held = shares_per_trade
                    entry_price = buy_price
                    entry_time = timestamps[i]
                    total_costs += cost

            current_position = signal

        # Update equity
        mark_to_market = cash + (shares_held * price)
        equity[i] = mark_to_market
        positions[i] = current_position

        # Track drawdown
        peak_equity = max(peak_equity, mark_to_market)
        drawdowns[i] = (peak_equity - mark_to_market) / peak_equity if peak_equity > 0 else 0

    # Build equity curve
    equity_curve = pd.DataFrame({
        "timestamp": timestamps,
        "equity": equity,
        "position": positions,
        "drawdown": drawdowns,
    })

    # Build trade log
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
        columns=["entry_time", "exit_time", "entry_price", "exit_price",
                 "shares", "pnl", "cost", "return_pct"]
    )

    return {
        "equity_curve": equity_curve,
        "trades": trades_df,
        "total_costs": total_costs,
        "initial_capital": initial_capital,
    }
