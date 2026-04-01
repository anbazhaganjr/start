"""
Reward functions for RL trading environment.

Implements after-cost reward with drawdown penalty for realistic training.
"""

import numpy as np

from start.utils.constants import SLIPPAGE_PCT, COMMISSION_PER_SHARE, RL_DRAWDOWN_PENALTY


def after_cost_reward(
    price_now: float,
    price_prev: float,
    position: int,
    action: int,
    shares: int = 100,
    slippage_pct: float = SLIPPAGE_PCT,
    commission_per_share: float = COMMISSION_PER_SHARE,
) -> float:
    """
    Compute reward including transaction costs.

    Args:
        price_now: Current bar close price.
        price_prev: Previous bar close price.
        position: Current position before action (0=flat, 1=long).
        action: Action taken (0=hold, 1=buy, 2=sell).
        shares: Position size.
        slippage_pct: Slippage as fraction of price.
        commission_per_share: Commission per share.

    Returns:
        Reward in dollar terms (normalized by position value).
    """
    # PnL from holding position
    if position == 1:
        pnl = (price_now - price_prev) / price_prev
    else:
        pnl = 0.0

    # Transaction costs on trades
    cost = 0.0
    if action == 1 and position == 0:  # Buy
        cost = slippage_pct + (commission_per_share * shares) / (price_now * shares)
    elif action == 2 and position == 1:  # Sell
        cost = slippage_pct + (commission_per_share * shares) / (price_now * shares)

    return pnl - cost


def drawdown_penalty(
    equity: float,
    peak_equity: float,
    penalty_weight: float = RL_DRAWDOWN_PENALTY,
) -> float:
    """
    Penalty term for drawdown to discourage excessive risk.

    Args:
        equity: Current portfolio equity.
        peak_equity: Maximum equity seen so far.
        penalty_weight: Weight of drawdown penalty.

    Returns:
        Negative penalty proportional to drawdown.
    """
    if peak_equity <= 0:
        return 0.0

    dd = (peak_equity - equity) / peak_equity
    return -penalty_weight * dd


def shaped_reward(
    price_now: float,
    price_prev: float,
    position: int,
    action: int,
    equity: float,
    peak_equity: float,
    shares: int = 100,
) -> float:
    """
    Combined reward: after-cost PnL + drawdown penalty.

    This is the primary reward function used by RL agents.
    """
    r = after_cost_reward(price_now, price_prev, position, action, shares)
    dd = drawdown_penalty(equity, peak_equity)
    return r + dd
