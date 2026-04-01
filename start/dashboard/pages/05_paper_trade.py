"""Paper Trade — Simulated live trading view."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import get_project_root
from start.data.storage import load_features
from start.features.builder import get_feature_columns
from start.models.baselines import buy_and_hold, ma_crossover, rsi_mean_reversion
from start.backtest.engine import backtest_signals
from start.backtest.metrics import compute_metrics

st.set_page_config(page_title="Paper Trade", page_icon="📊", layout="wide")
st.title("📊 Paper Trade Simulator")
st.markdown("""
> **What this page does:** Simulates trading with $100,000 of fake money using different strategies.
> Pick a stock, choose a strategy, and see how it would have performed. No real money is at risk.
""")

with st.expander("Strategy Explanations", expanded=False):
    st.markdown("""
    | Strategy | How It Works |
    |---|---|
    | **Buy & Hold** | Buy the stock on day 1 and never sell. The simplest benchmark. |
    | **MA Crossover** | Buy when the 20-day average crosses above the 50-day average (uptrend starting). Sell when it crosses below (downtrend starting). |
    | **RSI Mean Reversion** | Buy when RSI < 30 (stock is "oversold" / beaten down). Sell when RSI > 70 (stock is "overbought" / overheated). |
    """)

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
    symbol = st.selectbox("Symbol", available,
                          help="Pick any of the 12 high-liquidity stocks in our universe")
with col2:
    strategy_name = st.selectbox("Strategy", ["Buy & Hold", "MA Crossover", "RSI Mean Reversion"],
                                 help="Different trading approaches to compare")
with col3:
    lookback = st.slider("Lookback Bars", 100, 3000, 500,
                         help="How many recent data points to simulate on")

df = load_features(symbol, "1h")
if df.empty:
    st.warning(f"No data for {symbol}")
    st.stop()

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

# ──────────────────────────────────────────────────
# 1. KPI CARDS — styled with color
# ──────────────────────────────────────────────────
st.markdown("---")
c1, c2, c3, c4, c5, c6 = st.columns(6)

pnl_color = "normal" if metrics["net_pnl"] >= 0 else "inverse"
c1.metric("Net Profit/Loss", f"${metrics['net_pnl']:,.0f}",
          help="Total money made or lost after all costs")
c2.metric("Return", f"{metrics['total_return']:.1%}",
          help="Percentage return on the $100K starting capital")
c3.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}",
          help="Return per unit of risk. >1 is good, >2 is excellent")
c4.metric("Max Drawdown", f"{metrics['max_drawdown']:.1%}",
          help="The worst peak-to-trough loss (how scary it got)")
c5.metric("Win Rate", f"{metrics['win_rate']:.0%}",
          help="Percentage of trades that made money")
c6.metric("Total Trades", f"{metrics['n_trades']}",
          help="Number of complete buy-sell round trips")

# ──────────────────────────────────────────────────
# 2. EQUITY CURVE with buy/sell markers
# ──────────────────────────────────────────────────
st.header("Portfolio Value Over Time")
st.caption("Green shaded areas = in a position (holding stock). The line shows your portfolio value.")

timestamps = pd.to_datetime(eq["timestamp"])

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Equity Curve", "Position Status"),
    row_heights=[0.75, 0.25],
)

# Equity line
fig.add_trace(go.Scatter(
    x=timestamps, y=eq["equity"],
    name="Portfolio Value",
    line=dict(color="#2196F3", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(33, 150, 243, 0.08)",
    hovertemplate="Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>",
), row=1, col=1)

# Starting capital line
fig.add_hline(y=100000, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1,
              annotation_text="Starting Capital ($100K)")

# Peak equity line
peak_eq = eq["equity"].cummax()
fig.add_trace(go.Scatter(
    x=timestamps, y=peak_eq,
    name="Peak Value",
    line=dict(color="rgba(255, 152, 0, 0.4)", width=1, dash="dot"),
    hovertemplate="Peak: $%{y:,.0f}<extra></extra>",
), row=1, col=1)

# Buy/sell markers on equity curve
trades_df = bt["trades"]
if not trades_df.empty:
    buy_times = pd.to_datetime(trades_df["entry_time"])
    sell_times = pd.to_datetime(trades_df["exit_time"])

    # Find equity values at entry/exit times
    eq_ts = pd.to_datetime(eq["timestamp"])
    buy_eqs = []
    for bt_time in buy_times:
        idx = (eq_ts - bt_time).abs().idxmin()
        buy_eqs.append(eq["equity"].iloc[idx])
    sell_eqs = []
    for st_time in sell_times:
        idx = (eq_ts - st_time).abs().idxmin()
        sell_eqs.append(eq["equity"].iloc[idx])

    fig.add_trace(go.Scatter(
        x=buy_times, y=buy_eqs,
        mode="markers", name="Buy",
        marker=dict(symbol="triangle-up", size=10, color="#4CAF50", line=dict(width=1, color="white")),
        hovertemplate="BUY<br>Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=sell_times, y=sell_eqs,
        mode="markers", name="Sell",
        marker=dict(symbol="triangle-down", size=10, color="#F44336", line=dict(width=1, color="white")),
        hovertemplate="SELL<br>Date: %{x}<br>Equity: $%{y:,.0f}<extra></extra>",
    ), row=1, col=1)

# Position indicator (0 or 1)
pos_colors = ["#4CAF50" if p == 1 else "#EEEEEE" for p in eq["position"]]
fig.add_trace(go.Bar(
    x=timestamps, y=eq["position"],
    name="In Position",
    marker_color=pos_colors,
    opacity=0.6,
    hovertemplate="Position: %{y}<extra></extra>",
), row=2, col=1)

fig.update_layout(
    height=650,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    hovermode="x unified",
)
fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
fig.update_yaxes(title_text="Position", tickvals=[0, 1], ticktext=["Cash", "Invested"], row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────
# 3. TRADE ANALYSIS — PnL waterfall
# ──────────────────────────────────────────────────
if not trades_df.empty:
    st.header("Trade-by-Trade Analysis")
    st.caption("Each bar shows the profit or loss from one trade. Green = profitable, Red = losing trade.")

    trade_display = trades_df.copy()
    trade_display["trade_num"] = range(1, len(trade_display) + 1)
    trade_display["label"] = trade_display["trade_num"].apply(lambda x: f"Trade {x}")

    colors = ["#4CAF50" if p >= 0 else "#F44336" for p in trade_display["pnl"]]

    fig_trades = go.Figure()
    fig_trades.add_trace(go.Bar(
        x=trade_display["label"],
        y=trade_display["pnl"],
        marker_color=colors,
        text=[f"${p:,.0f}" for p in trade_display["pnl"]],
        textposition="outside",
        textfont=dict(size=9),
        hovertemplate=(
            "Trade #%{customdata[0]}<br>"
            "Entry: $%{customdata[1]:.2f}<br>"
            "Exit: $%{customdata[2]:.2f}<br>"
            "PnL: $%{y:,.2f}<br>"
            "Return: %{customdata[3]:.2%}<extra></extra>"
        ),
        customdata=trade_display[["trade_num", "entry_price", "exit_price", "return_pct"]].values,
    ))

    fig_trades.add_hline(y=0, line_color="gray", line_width=1)
    fig_trades.update_layout(
        title="Individual Trade Results",
        yaxis_title="Profit/Loss ($)",
        height=400,
    )

    # Only show x labels if not too many trades
    if len(trade_display) > 30:
        fig_trades.update_xaxes(showticklabels=False)

    st.plotly_chart(fig_trades, use_container_width=True)

    # Cumulative PnL
    trade_display["cumulative_pnl"] = trade_display["pnl"].cumsum()

    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=trade_display["label"],
        y=trade_display["cumulative_pnl"],
        mode="lines+markers",
        line=dict(color="#2196F3", width=2.5),
        marker=dict(size=6, color=colors, line=dict(width=1, color="white")),
        fill="tozeroy",
        fillcolor="rgba(33, 150, 243, 0.1)",
        hovertemplate="Trade #%{customdata}<br>Cumulative PnL: $%{y:,.0f}<extra></extra>",
        customdata=trade_display["trade_num"],
    ))
    fig_cum.add_hline(y=0, line_color="gray", line_dash="dash")
    fig_cum.update_layout(
        title="Cumulative Profit/Loss Over Trades",
        yaxis_title="Cumulative PnL ($)",
        height=350,
    )
    if len(trade_display) > 30:
        fig_cum.update_xaxes(showticklabels=False)
    st.plotly_chart(fig_cum, use_container_width=True)

    # Win/Loss donut
    col1, col2 = st.columns(2)
    with col1:
        wins = (trade_display["pnl"] > 0).sum()
        losses = (trade_display["pnl"] <= 0).sum()
        fig_wl = go.Figure(data=[go.Pie(
            labels=["Wins", "Losses"],
            values=[wins, losses],
            marker=dict(colors=["#4CAF50", "#F44336"]),
            hole=0.5,
            textinfo="label+value+percent",
        )])
        fig_wl.update_layout(
            title="Win/Loss Breakdown",
            height=300,
            annotations=[dict(text=f"{wins}/{wins+losses}", x=0.5, y=0.5, font_size=18, showarrow=False)],
        )
        st.plotly_chart(fig_wl, use_container_width=True)

    with col2:
        avg_win = trade_display[trade_display["pnl"] > 0]["pnl"].mean() if wins > 0 else 0
        avg_loss = trade_display[trade_display["pnl"] <= 0]["pnl"].mean() if losses > 0 else 0

        fig_avgs = go.Figure()
        fig_avgs.add_trace(go.Bar(
            x=["Average Win", "Average Loss"],
            y=[avg_win, avg_loss],
            marker_color=["#4CAF50", "#F44336"],
            text=[f"${avg_win:,.0f}", f"${avg_loss:,.0f}"],
            textposition="outside",
        ))
        fig_avgs.update_layout(title="Average Win vs Loss", height=300, yaxis_title="Dollars ($)")
        st.plotly_chart(fig_avgs, use_container_width=True)

    # Trade log table
    with st.expander("View Full Trade Log", expanded=False):
        display_trades = trades_df.copy()
        display_trades["entry_time"] = pd.to_datetime(display_trades["entry_time"])
        display_trades["exit_time"] = pd.to_datetime(display_trades["exit_time"])
        st.dataframe(
            display_trades.style.format({
                "entry_price": "${:.2f}", "exit_price": "${:.2f}",
                "pnl": "${:.2f}", "cost": "${:.4f}", "return_pct": "{:.2%}",
            }),
            use_container_width=True,
        )
else:
    st.info("No trades executed with this strategy (Buy & Hold holds the entire time).")

# ──────────────────────────────────────────────────
# 4. DETAILED METRICS TABLE
# ──────────────────────────────────────────────────
st.header("Full Performance Breakdown")
st.caption("All the risk and return metrics for this simulation.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Returns**")
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Net PnL | ${metrics['net_pnl']:,.2f} |
    | Total Return | {metrics['total_return']:.2%} |
    | Annualized Return | {metrics['annualized_return']:.2%} |
    | Final Equity | ${metrics['final_equity']:,.2f} |
    """)
with col2:
    st.markdown("**Risk**")
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Sharpe Ratio | {metrics['sharpe_ratio']:.3f} |
    | Sortino Ratio | {metrics['sortino_ratio']:.3f} |
    | Calmar Ratio | {metrics['calmar_ratio']:.3f} |
    | Max Drawdown | {metrics['max_drawdown']:.2%} |
    | Volatility | {metrics['volatility']:.4f} |
    """)
with col3:
    st.markdown("**Trading**")
    st.markdown(f"""
    | Metric | Value |
    |--------|-------|
    | Win Rate | {metrics['win_rate']:.2%} |
    | Profit Factor | {metrics['profit_factor']:.3f} |
    | Avg Win | ${metrics['avg_win']:,.2f} |
    | Avg Loss | ${metrics['avg_loss']:,.2f} |
    | Total Costs | ${metrics['total_costs']:,.2f} |
    | Turnover | {metrics['turnover']:.4f} |
    """)
