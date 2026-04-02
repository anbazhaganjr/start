"""PnL Charts — Equity curves and drawdown analysis."""

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
from start.data.storage import load_features, load_results
from start.features.builder import get_feature_columns
from start.models.baselines import buy_and_hold, ma_crossover
from start.backtest.engine import backtest_signals
from start.backtest.metrics import compute_metrics
from start.dashboard.components import page_footer

st.set_page_config(page_title="PnL Charts", page_icon="💰", layout="wide")
st.title("💰 PnL & Equity Analysis")
st.markdown("""
> **What this page shows:** How different trading strategies perform over time on real stock data.
> An *equity curve* tracks the value of a $100,000 portfolio. *Drawdown* shows the worst peak-to-trough drops.
""")

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

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Data Points:** {len(df):,} bars")
st.sidebar.markdown(f"**Features:** {len(get_feature_columns(df))} indicators")
st.sidebar.markdown(f"**Date Range:** {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Generate signals and backtest
strategies = {}
for name, func in [("Buy & Hold", buy_and_hold), ("MA Crossover", ma_crossover)]:
    signals = func(df)
    bt = backtest_signals(df, signals)
    strategies[name] = bt

# ──────────────────────────────────────────────────
# 1. EQUITY CURVE + DRAWDOWN (multi-panel)
# ──────────────────────────────────────────────────
st.header("Equity Curve Comparison")
st.caption("The equity curve shows how a $100K investment grows (or shrinks) over time using each strategy.")

fig = make_subplots(
    rows=2, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.08,
    subplot_titles=("Portfolio Value Over Time", "Drawdown (how far below the peak)"),
    row_heights=[0.65, 0.35],
)

colors = {"Buy & Hold": "#2196F3", "MA Crossover": "#FF9800"}

for name, bt in strategies.items():
    eq = bt["equity_curve"]
    timestamps = pd.to_datetime(eq["timestamp"])

    fig.add_trace(
        go.Scatter(
            x=timestamps, y=eq["equity"],
            name=name,
            line=dict(color=colors.get(name, "#666"), width=2.5),
            hovertemplate=f"{name}<br>Date: %{{x}}<br>Equity: $%{{y:,.0f}}<extra></extra>",
        ),
        row=1, col=1,
    )
    # Filled drawdown
    fig.add_trace(
        go.Scatter(
            x=timestamps, y=-eq["drawdown"] * 100,
            name=f"{name} Drawdown",
            fill="tozeroy",
            line=dict(color=colors.get(name, "#666"), width=1),
            fillcolor=colors.get(name, "#666").replace(")", ", 0.2)").replace("rgb", "rgba") if "rgb" in colors.get(name, "") else f"rgba(100,100,100,0.2)",
            hovertemplate=f"{name}<br>Drawdown: %{{y:.1f}}%<extra></extra>",
        ),
        row=2, col=1,
    )

fig.update_layout(
    height=650,
    yaxis_title="Portfolio Value ($)",
    yaxis2_title="Drawdown (%)",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified",
)
fig.add_hline(y=100000, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1,
              annotation_text="Starting Capital", annotation_position="bottom right")
st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────
# 2. KPI CARDS WITH COMPARISON
# ──────────────────────────────────────────────────
st.header("Performance Scorecard")
st.caption("Key metrics comparing both strategies. Green/red arrows show which strategy is better.")

metrics_data = []
for name, bt in strategies.items():
    m = compute_metrics(bt)
    m["strategy"] = name
    metrics_data.append(m)

metrics_df = pd.DataFrame(metrics_data).set_index("strategy")

cols = st.columns(4)
metric_labels = [
    ("net_pnl", "Net Profit/Loss", "${:,.0f}", "How much money was made or lost"),
    ("sharpe_ratio", "Sharpe Ratio", "{:.3f}", "Return per unit of risk (higher = better)"),
    ("max_drawdown", "Max Drawdown", "{:.1%}", "Worst peak-to-trough loss (lower = better)"),
    ("win_rate", "Win Rate", "{:.1%}", "Percentage of profitable trades"),
]

for i, (key, label, fmt, tooltip) in enumerate(metric_labels):
    if key in metrics_df.columns:
        with cols[i]:
            bh = metrics_df.loc["Buy & Hold", key] if "Buy & Hold" in metrics_df.index else 0
            ma = metrics_df.loc["MA Crossover", key] if "MA Crossover" in metrics_df.index else 0
            delta_val = ma - bh
            delta_str = fmt.format(delta_val) if delta_val != 0 else None
            inv = "inverse" if key == "max_drawdown" else "normal"
            st.metric(label, fmt.format(ma), delta=delta_str, delta_color=inv,
                      help=tooltip)

# Full metrics table
with st.expander("View All Metrics", expanded=False):
    display_cols = ["net_pnl", "total_return", "sharpe_ratio", "sortino_ratio",
                    "calmar_ratio", "max_drawdown", "win_rate", "n_trades", "profit_factor"]
    display_cols = [c for c in display_cols if c in metrics_df.columns]
    st.dataframe(
        metrics_df[display_cols].style.format({
            "net_pnl": "${:,.2f}", "total_return": "{:.2%}", "sharpe_ratio": "{:.3f}",
            "sortino_ratio": "{:.3f}", "calmar_ratio": "{:.3f}",
            "max_drawdown": "{:.2%}", "win_rate": "{:.2%}",
            "n_trades": "{:.0f}", "profit_factor": "{:.3f}",
        }),
        use_container_width=True,
    )

# ──────────────────────────────────────────────────
# 3. INTERACTIVE CANDLESTICK + INDICATORS
# ──────────────────────────────────────────────────
st.header("Price Action & Technical Indicators")
st.caption("""
**Candlesticks** show open/high/low/close prices. Green = price went up, Red = price went down.
**SMA** (Simple Moving Average) smooths price to show the trend. **RSI** measures momentum (overbought > 70, oversold < 30).
""")

# Let user pick a date range
n_bars = len(df)
range_options = {"Last 100 bars": 100, "Last 500 bars": 500, "Last 1000 bars": 1000, "All data": n_bars}
selected_range = st.radio("Time Range", list(range_options.keys()), index=1, horizontal=True)
view_df = df.tail(range_options[selected_range]).reset_index(drop=True)
timestamps = pd.to_datetime(view_df["timestamp"])

fig2 = make_subplots(
    rows=4, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.45, 0.18, 0.18, 0.19],
    subplot_titles=("Price & Moving Averages", "RSI (Momentum)", "MACD (Trend)", "Volume"),
)

# Candlestick
fig2.add_trace(
    go.Candlestick(
        x=timestamps, open=view_df["open"], high=view_df["high"],
        low=view_df["low"], close=view_df["close"],
        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ),
    row=1, col=1,
)

# Bollinger Bands as shaded area (read pre-computed columns)
if "bb_upper" in view_df.columns and "bb_lower" in view_df.columns:
    fig2.add_trace(go.Scatter(x=timestamps, y=view_df["bb_upper"], name="BB Upper",
                              line=dict(color="rgba(150,150,150,0.3)", width=0.5)), row=1, col=1)
    fig2.add_trace(go.Scatter(x=timestamps, y=view_df["bb_lower"], name="BB Lower",
                              line=dict(color="rgba(150,150,150,0.3)", width=0.5),
                              fill="tonexty", fillcolor="rgba(150,150,220,0.08)"), row=1, col=1)

if "sma_20" in view_df.columns:
    fig2.add_trace(go.Scatter(x=timestamps, y=view_df["sma_20"], name="SMA 20",
                              line=dict(color="#FF9800", width=1.5)), row=1, col=1)
if "sma_50" in view_df.columns:
    fig2.add_trace(go.Scatter(x=timestamps, y=view_df["sma_50"], name="SMA 50",
                              line=dict(color="#E91E63", width=1.5, dash="dash")), row=1, col=1)

# RSI
rsi_col = "rsi_14" if "rsi_14" in view_df.columns else "rsi" if "rsi" in view_df.columns else None
if rsi_col:
    rsi_colors = ["#26a69a" if v < 30 else "#ef5350" if v > 70 else "#78909C" for v in view_df[rsi_col].fillna(50)]
    fig2.add_trace(go.Scatter(x=timestamps, y=view_df[rsi_col], name="RSI",
                              line=dict(color="#AB47BC", width=1.5)), row=2, col=1)
    fig2.add_hline(y=70, line_dash="dash", line_color="#ef5350", opacity=0.6, row=2, col=1,
                   annotation_text="Overbought (70)")
    fig2.add_hline(y=30, line_dash="dash", line_color="#26a69a", opacity=0.6, row=2, col=1,
                   annotation_text="Oversold (30)")
    fig2.add_hrect(y0=30, y1=70, fillcolor="rgba(200,200,200,0.05)", line_width=0, row=2, col=1)

# MACD
if "macd" in view_df.columns and "macd_hist" in view_df.columns:
    macd_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in view_df["macd_hist"].fillna(0)]
    fig2.add_trace(go.Bar(x=timestamps, y=view_df["macd_hist"], name="MACD Histogram",
                          marker_color=macd_colors), row=3, col=1)
    fig2.add_trace(go.Scatter(x=timestamps, y=view_df["macd"], name="MACD Line",
                              line=dict(color="#2196F3", width=1.5)), row=3, col=1)

# Volume with color
if "volume" in view_df.columns:
    vol_colors = ["#26a69a" if view_df["close"].iloc[i] >= view_df["open"].iloc[i] else "#ef5350"
                  for i in range(len(view_df))]
    fig2.add_trace(
        go.Bar(x=timestamps, y=view_df["volume"], name="Volume",
               marker_color=vol_colors, opacity=0.6),
        row=4, col=1,
    )

fig2.update_layout(
    height=900,
    xaxis_rangeslider_visible=False,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    hovermode="x unified",
)
fig2.update_yaxes(title_text="Price ($)", row=1, col=1)
fig2.update_yaxes(title_text="RSI", row=2, col=1)
fig2.update_yaxes(title_text="MACD", row=3, col=1)
fig2.update_yaxes(title_text="Volume", row=4, col=1)

st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────────────────────────
# 4. ROLLING VOLATILITY CHART
# ──────────────────────────────────────────────────
if "rolling_volatility" in view_df.columns:
    st.header("Volatility Over Time")
    st.caption("Volatility measures how wildly the price swings. Spikes often occur during market stress or earnings.")

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=timestamps,
        y=view_df["rolling_volatility"] * 100,
        fill="tozeroy",
        fillcolor="rgba(255, 152, 0, 0.15)",
        line=dict(color="#FF9800", width=1.5),
        hovertemplate="Volatility: %{y:.2f}%<extra></extra>",
    ))
    fig_vol.update_layout(
        title=f"{symbol} -- 5-Bar Rolling Volatility",
        yaxis_title="Volatility (%)",
        height=350,
        hovermode="x unified",
    )
    st.plotly_chart(fig_vol, use_container_width=True)

# ──────────────────────────────────────────────────
# 5. RETURNS DISTRIBUTION
# ──────────────────────────────────────────────────
if "simple_return" in df.columns:
    st.header("Return Distribution")
    st.caption("Shows how returns are distributed. A 'normal' bell curve centered near 0 is typical. Fat tails mean extreme moves are more common than expected.")

    returns = df["simple_return"].dropna()
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=returns * 100,
        nbinsx=80,
        marker_color="#2196F3",
        opacity=0.7,
        name="Returns",
        hovertemplate="Return: %{x:.2f}%<br>Count: %{y}<extra></extra>",
    ))
    # Add normal distribution overlay
    x_range = np.linspace(returns.min() * 100, returns.max() * 100, 200)
    from scipy.stats import norm
    try:
        mu, std = returns.mean() * 100, returns.std() * 100
        normal_y = norm.pdf(x_range, mu, std) * len(returns) * (returns.max() - returns.min()) * 100 / 80
        fig_dist.add_trace(go.Scatter(
            x=x_range, y=normal_y,
            name="Normal Distribution",
            line=dict(color="#FF9800", width=2, dash="dash"),
        ))
    except Exception:
        pass

    fig_dist.update_layout(
        title=f"{symbol} -- Return Distribution (bar interval = {interval})",
        xaxis_title="Return (%)",
        yaxis_title="Frequency",
        height=400,
        bargap=0.05,
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Return", f"{returns.mean():.4%}", help="Average return per bar")
    c2.metric("Std Dev", f"{returns.std():.4%}", help="Standard deviation of returns")
    c3.metric("Skewness", f"{returns.skew():.3f}", help="Negative = more extreme losses than gains")
    c4.metric("Kurtosis", f"{returns.kurtosis():.3f}", help="Higher = fatter tails (more extreme events)")

# ──────────────────────────────────────────────────
# DOWNLOAD BUTTON
# ──────────────────────────────────────────────────
st.download_button(
    label="Download feature data as CSV",
    data=df.to_csv(index=False),
    file_name=f"{symbol}_{interval}_features.csv",
    mime="text/csv",
)

page_footer()
