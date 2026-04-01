"""PnL Charts — Equity curves and drawdown analysis."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import get_project_root
from start.data.storage import load_features, load_results
from start.features.builder import get_feature_columns
from start.models.baselines import buy_and_hold, ma_crossover
from start.backtest.engine import backtest_signals
from start.backtest.metrics import compute_metrics

st.set_page_config(page_title="PnL Charts", page_icon="💰", layout="wide")
st.title("💰 PnL & Equity Analysis")

root = get_project_root()

# Symbol selector
features_dir = root / "data" / "features"
available = sorted([f.stem.replace("_1h", "").replace("_1d", "")
                    for f in features_dir.glob("*.parquet")]) if features_dir.exists() else []

if not available:
    st.warning("No feature files found. Run scripts 01-02 first.")
    st.stop()

symbol = st.sidebar.selectbox("Symbol", available, index=available.index("AAPL") if "AAPL" in available else 0)
interval = st.sidebar.selectbox("Interval", ["1h", "1d"], index=0)

df = load_features(symbol, interval)
if df.empty:
    st.warning(f"No data for {symbol} ({interval})")
    st.stop()

st.sidebar.markdown(f"**Bars:** {len(df):,}")
st.sidebar.markdown(f"**Features:** {len(get_feature_columns(df))}")

# Generate signals and backtest
strategies = {}
for name, func in [("Buy & Hold", buy_and_hold), ("MA Crossover", ma_crossover)]:
    signals = func(df)
    bt = backtest_signals(df, signals)
    strategies[name] = bt

# Equity curve chart
fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    subplot_titles=("Equity Curve", "Drawdown"),
    row_heights=[0.7, 0.3],
)

colors = {"Buy & Hold": "#2196F3", "MA Crossover": "#FF9800"}

for name, bt in strategies.items():
    eq = bt["equity_curve"]
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(eq["timestamp"]),
            y=eq["equity"],
            name=name,
            line=dict(color=colors.get(name, "#666")),
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pd.to_datetime(eq["timestamp"]),
            y=-eq["drawdown"] * 100,
            name=f"{name} DD",
            fill="tozeroy",
            line=dict(color=colors.get(name, "#666"), dash="dot"),
        ),
        row=2, col=1,
    )

fig.update_layout(
    height=700,
    title=f"{symbol} — Baseline Equity Curves",
    yaxis_title="Equity ($)",
    yaxis2_title="Drawdown (%)",
    showlegend=True,
)
st.plotly_chart(fig, use_container_width=True)

# Metrics table
st.subheader("Performance Metrics")
metrics_data = []
for name, bt in strategies.items():
    m = compute_metrics(bt)
    m["strategy"] = name
    metrics_data.append(m)

metrics_df = pd.DataFrame(metrics_data).set_index("strategy")
display_cols = ["net_pnl", "total_return", "sharpe_ratio", "sortino_ratio",
                "max_drawdown", "win_rate", "n_trades", "profit_factor"]
display_cols = [c for c in display_cols if c in metrics_df.columns]

st.dataframe(
    metrics_df[display_cols].style.format({
        "net_pnl": "${:,.2f}",
        "total_return": "{:.2%}",
        "sharpe_ratio": "{:.3f}",
        "sortino_ratio": "{:.3f}",
        "max_drawdown": "{:.2%}",
        "win_rate": "{:.2%}",
        "n_trades": "{:.0f}",
        "profit_factor": "{:.3f}",
    }),
    use_container_width=True,
)

# Price chart with indicators
st.subheader("Price & Indicators")
fig2 = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.5, 0.25, 0.25],
)

timestamps = pd.to_datetime(df["timestamp"])

fig2.add_trace(
    go.Candlestick(
        x=timestamps,
        open=df["open"], high=df["high"],
        low=df["low"], close=df["close"],
        name="OHLC",
    ),
    row=1, col=1,
)

if "sma_20" in df.columns:
    fig2.add_trace(go.Scatter(x=timestamps, y=df["sma_20"], name="SMA20",
                              line=dict(color="orange", width=1)), row=1, col=1)
if "sma_50" in df.columns:
    fig2.add_trace(go.Scatter(x=timestamps, y=df["sma_50"], name="SMA50",
                              line=dict(color="blue", width=1)), row=1, col=1)
if "rsi" in df.columns:
    fig2.add_trace(go.Scatter(x=timestamps, y=df["rsi"], name="RSI",
                              line=dict(color="purple")), row=2, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig2.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

fig2.add_trace(
    go.Bar(x=timestamps, y=df["volume"], name="Volume",
           marker_color="rgba(100,100,200,0.3)"),
    row=3, col=1,
)

fig2.update_layout(height=800, title=f"{symbol} — Price & Technical Indicators",
                   xaxis_rangeslider_visible=False)
st.plotly_chart(fig2, use_container_width=True)
