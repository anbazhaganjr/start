"""Paper Trade — Simulated live trading view."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import get_project_root
from start.data.storage import load_features
from start.features.builder import get_feature_columns
from start.models.baselines import buy_and_hold, ma_crossover, rsi_mean_reversion
from start.backtest.engine import backtest_signals
from start.backtest.metrics import compute_metrics

st.set_page_config(page_title="Paper Trade", page_icon="📊", layout="wide")
st.title("📊 Paper Trade Simulator")
st.markdown("Simulate strategy execution on recent data.")

root = get_project_root()

# Controls
features_dir = root / "data" / "features"
available = sorted([f.stem.replace("_1h", "").replace("_1d", "")
                    for f in features_dir.glob("*.parquet")]) if features_dir.exists() else []

if not available:
    st.warning("No feature files found.")
    st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("Symbol", available)
with col2:
    strategy_name = st.selectbox("Strategy", ["Buy & Hold", "MA Crossover", "RSI Mean Reversion"])
with col3:
    lookback = st.slider("Lookback Bars", 100, 3000, 500)

df = load_features(symbol, "1h")
if df.empty:
    st.warning(f"No data for {symbol}")
    st.stop()

# Use last N bars
df = df.tail(lookback).reset_index(drop=True)

# Generate signals
strategy_map = {
    "Buy & Hold": buy_and_hold,
    "MA Crossover": ma_crossover,
    "RSI Mean Reversion": rsi_mean_reversion,
}

signals = strategy_map[strategy_name](df)
bt = backtest_signals(df, signals)
metrics = compute_metrics(bt)
eq = bt["equity_curve"]

# KPI cards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Net PnL", f"${metrics['net_pnl']:,.2f}")
c2.metric("Return", f"{metrics['total_return']:.2%}")
c3.metric("Sharpe", f"{metrics['sharpe_ratio']:.3f}")
c4.metric("Max DD", f"{metrics['max_drawdown']:.2%}")
c5.metric("Trades", f"{metrics['n_trades']}")

# Equity curve with position overlay
fig = go.Figure()

timestamps = pd.to_datetime(eq["timestamp"])

fig.add_trace(go.Scatter(
    x=timestamps, y=eq["equity"],
    name="Equity", line=dict(color="#2196F3", width=2),
))

# Color background by position
for i in range(1, len(eq)):
    if eq["position"].iloc[i] == 1:
        fig.add_vrect(
            x0=timestamps.iloc[i-1], x1=timestamps.iloc[i],
            fillcolor="rgba(76, 175, 80, 0.1)",
            layer="below", line_width=0,
        )

fig.update_layout(
    title=f"{symbol} — {strategy_name} Paper Trade",
    yaxis_title="Equity ($)",
    height=500,
)
st.plotly_chart(fig, use_container_width=True)

# Trade log
trades_df = bt["trades"]
if not trades_df.empty:
    st.subheader("Trade Log")
    display_trades = trades_df.copy()
    display_trades["entry_time"] = pd.to_datetime(display_trades["entry_time"])
    display_trades["exit_time"] = pd.to_datetime(display_trades["exit_time"])
    st.dataframe(
        display_trades.style.format({
            "entry_price": "${:.2f}",
            "exit_price": "${:.2f}",
            "pnl": "${:.2f}",
            "cost": "${:.4f}",
            "return_pct": "{:.2%}",
        }),
        use_container_width=True,
    )
else:
    st.info("No trades executed.")

# Additional metrics
st.subheader("Detailed Metrics")
col1, col2 = st.columns(2)
with col1:
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Win Rate | {metrics['win_rate']:.2%} |
    | Profit Factor | {metrics['profit_factor']:.3f} |
    | Avg Win | ${metrics['avg_win']:,.2f} |
    | Avg Loss | ${metrics['avg_loss']:,.2f} |
    | Sortino | {metrics['sortino_ratio']:.3f} |
    | Calmar | {metrics['calmar_ratio']:.3f} |
    """)
with col2:
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Total Costs | ${metrics['total_costs']:,.2f} |
    | Turnover | {metrics['turnover']:.4f} |
    | Volatility | {metrics['volatility']:.4f} |
    | Final Equity | ${metrics['final_equity']:,.2f} |
    | N Bars | {metrics['n_bars']:,} |
    | Annualized Return | {metrics['annualized_return']:.2%} |
    """)
