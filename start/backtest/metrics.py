"""
Performance metrics for backtesting.

Computes risk-adjusted returns, drawdown analysis, and trade statistics.
"""

import numpy as np
import pandas as pd

from start.utils.constants import ANNUALIZATION_FACTOR
from start.utils.logger import get_logger

logger = get_logger(__name__)


def compute_metrics(backtest_result: dict, annualize: bool = True) -> dict:
    """
    Compute comprehensive performance metrics from backtest results.

    Args:
        backtest_result: Dict from backtest_signals().
        annualize: Whether to annualize return and ratio metrics.

    Returns:
        Dict of named metrics.
    """
    eq = backtest_result["equity_curve"]
    trades_df = backtest_result["trades"]
    initial = backtest_result["initial_capital"]

    equity = eq["equity"].values
    n_bars = len(equity)

    # --- Returns ---
    returns = pd.Series(equity).pct_change().dropna()
    total_return = (equity[-1] - initial) / initial if initial > 0 else 0
    net_pnl = equity[-1] - initial

    # Annualization
    ann_factor = ANNUALIZATION_FACTOR if annualize else 1.0

    # --- Sharpe Ratio ---
    if returns.std() > 0:
        sharpe = (returns.mean() / returns.std()) * np.sqrt(ann_factor)
    else:
        sharpe = 0.0

    # --- Sortino Ratio (downside deviation only) ---
    downside = returns[returns < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (returns.mean() / downside.std()) * np.sqrt(ann_factor)
    else:
        sortino = 0.0

    # --- Max Drawdown ---
    max_dd = eq["drawdown"].max()

    # --- Calmar Ratio ---
    ann_return = total_return * (ann_factor / n_bars) if n_bars > 0 else 0
    calmar = ann_return / max_dd if max_dd > 0 else 0.0

    # --- Trade Statistics ---
    n_trades = len(trades_df)
    if n_trades > 0:
        winning = trades_df[trades_df["pnl"] > 0]
        losing = trades_df[trades_df["pnl"] <= 0]
        win_rate = len(winning) / n_trades
        avg_win = winning["pnl"].mean() if len(winning) > 0 else 0
        avg_loss = losing["pnl"].mean() if len(losing) > 0 else 0
        gross_profit = winning["pnl"].sum() if len(winning) > 0 else 0
        gross_loss = abs(losing["pnl"].sum()) if len(losing) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_trade_return = trades_df["return_pct"].mean()
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0
        avg_trade_return = 0

    # --- Turnover ---
    position_changes = (eq["position"].diff().abs() > 0).sum()
    turnover = position_changes / n_bars if n_bars > 0 else 0

    metrics = {
        # Profitability
        "net_pnl": net_pnl,
        "total_return": total_return,
        "annualized_return": ann_return,
        # Risk-adjusted
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        # Risk
        "max_drawdown": max_dd,
        "volatility": returns.std() * np.sqrt(ann_factor) if len(returns) > 0 else 0,
        # Trade stats
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "avg_trade_return": avg_trade_return,
        # Activity
        "turnover": turnover,
        "total_costs": backtest_result["total_costs"],
        # Data
        "n_bars": n_bars,
        "final_equity": equity[-1] if len(equity) > 0 else initial,
    }

    return metrics


def format_metrics(metrics: dict) -> str:
    """Format metrics as a readable string."""
    lines = [
        f"  Net PnL:          ${metrics['net_pnl']:>12,.2f}",
        f"  Total Return:     {metrics['total_return']:>12.2%}",
        f"  Sharpe Ratio:     {metrics['sharpe_ratio']:>12.3f}",
        f"  Sortino Ratio:    {metrics['sortino_ratio']:>12.3f}",
        f"  Calmar Ratio:     {metrics['calmar_ratio']:>12.3f}",
        f"  Max Drawdown:     {metrics['max_drawdown']:>12.2%}",
        f"  Win Rate:         {metrics['win_rate']:>12.2%}",
        f"  Profit Factor:    {metrics['profit_factor']:>12.3f}",
        f"  Trades:           {metrics['n_trades']:>12d}",
        f"  Total Costs:      ${metrics['total_costs']:>12,.2f}",
        f"  Final Equity:     ${metrics['final_equity']:>12,.2f}",
    ]
    return "\n".join(lines)


def compare_strategies(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a comparison table of multiple strategy metrics.

    Args:
        results: Dict of {strategy_name: metrics_dict}.

    Returns:
        DataFrame with strategies as rows and metrics as columns.
    """
    rows = []
    for name, metrics in results.items():
        row = {"strategy": name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("strategy")

    # Sort by Sharpe ratio descending
    df = df.sort_values("sharpe_ratio", ascending=False)

    return df
