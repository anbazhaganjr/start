"""Signal Heatmap — Model signal agreement across symbols."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import get_project_root
from start.data.storage import load_results
from start.dashboard.components import page_footer

st.set_page_config(page_title="Signal Heatmap", page_icon="🔥", layout="wide")
st.title("🔥 Signal Heatmap")
st.markdown("""
> **What this page shows:** We tested multiple trading strategies (simple rules, machine learning, and AI agents)
> across 12 major stocks. This page compares how each strategy performed on every stock using key financial metrics.
""")

with st.expander("Metric Glossary", expanded=False):
    st.markdown("""
    | Metric | What It Means | Good Value |
    |---|---|---|
    | **Sharpe Ratio** | Return earned per unit of risk taken | > 1.0 (> 2.0 is excellent) |
    | **Total Return** | Percentage gain/loss on the investment | Higher is better |
    | **Win Rate** | % of trades that made money | > 50% |
    | **Max Drawdown** | Worst peak-to-trough loss | Lower is better (less scary) |
    | **Sortino Ratio** | Like Sharpe but only penalizes downside risk | > 1.0 |
    | **Profit Factor** | Gross profit / Gross loss | > 1.0 means profitable |
    """)

root = get_project_root()

try:
    results = load_results("model_comparison")
    if results.empty:
        raise FileNotFoundError
except Exception:
    st.warning("No model comparison results found. Run script 03 first.")
    st.stop()

# Also load RL results
try:
    rl_results = load_results("rl_comparison")
except Exception:
    rl_results = pd.DataFrame()

combined = pd.concat([results, rl_results], ignore_index=True) if not rl_results.empty else results

# ──────────────────────────────────────────────────
# 1. RISK vs RETURN BUBBLE CHART
# ──────────────────────────────────────────────────
st.header("Risk vs Return Landscape")

bubble_size = combined["total_return"].clip(lower=0.001).abs() * 2000
fig_scatter = px.scatter(
    combined,
    x="max_drawdown",
    y="sharpe_ratio",
    color="strategy",
    size=bubble_size,
    hover_name="symbol",
    hover_data={"total_return": ":.2%", "win_rate": ":.1%", "n_trades": True},
    title="Risk-Return Tradeoff (bubble size = return magnitude)",
    labels={"max_drawdown": "Max Drawdown", "sharpe_ratio": "Sharpe Ratio"},
    color_discrete_sequence=px.colors.qualitative.Bold,
)
fig_scatter.update_layout(
    height=550,
    xaxis_tickformat=".1%",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
)
fig_scatter.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.85))
st.plotly_chart(fig_scatter, use_container_width=True)

# ──────────────────────────────────────────────────
# 2. ANNOTATED HEATMAP
# ──────────────────────────────────────────────────
st.header("Performance Heatmap")

metrics_to_show = st.selectbox(
    "Metric",
    ["sharpe_ratio", "total_return", "win_rate", "sortino_ratio", "calmar_ratio",
     "max_drawdown", "profit_factor", "n_trades"],
    index=0,
)

if "symbol" in combined.columns and "strategy" in combined.columns:
    pivot = combined.pivot_table(
        index="symbol", columns="strategy", values=metrics_to_show, aggfunc="first",
    )

    if not pivot.empty:
        fmt = ".2%" if metrics_to_show in ["total_return", "win_rate", "max_drawdown"] else ".3f"
        text_display = []
        for row in pivot.values:
            text_display.append([f"{v:{fmt}}" if not np.isnan(v) else "" for v in row])

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=text_display,
            texttemplate="%{text}",
            textfont={"size": 11},
            colorscale="RdYlGn" if metrics_to_show != "max_drawdown" else "RdYlGn_r",
            hovertemplate="Symbol: %{y}<br>Strategy: %{x}<br>Value: %{z:.4f}<extra></extra>",
        ))
        fig_heat.update_layout(
            title=f"{metrics_to_show.replace('_', ' ').title()} -- Symbol x Strategy",
            height=max(450, len(pivot) * 48),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ──────────────────────────────────────────────────
# 3. HORIZONTAL BAR RANKING
# ──────────────────────────────────────────────────
st.header("Strategy Ranking")

if "strategy" in combined.columns:
    avg_sorted = (
        combined.groupby("strategy")[metrics_to_show]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )

    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        x=avg_sorted[metrics_to_show],
        y=avg_sorted["strategy"],
        orientation="h",
        marker=dict(
            color=avg_sorted[metrics_to_show],
            colorscale="Viridis",
            line=dict(color="white", width=1.5),
        ),
        text=[f"{v:.4f}" for v in avg_sorted[metrics_to_show]],
        textposition="outside",
        hovertemplate="Strategy: %{y}<br>Value: %{x:.4f}<extra></extra>",
    ))
    fig_rank.update_layout(
        title=f"Average {metrics_to_show.replace('_', ' ').title()} by Strategy",
        height=400,
        xaxis_title=metrics_to_show.replace("_", " ").title(),
        margin=dict(l=150),
    )
    st.plotly_chart(fig_rank, use_container_width=True)

# ──────────────────────────────────────────────────
# 4. TREEMAP: Strategy -> Symbol hierarchy
# ──────────────────────────────────────────────────
st.header("Strategy-Symbol Treemap")

tree_data = combined.copy()
tree_data["abs_return"] = tree_data["total_return"].abs().clip(lower=0.001)
if not tree_data.empty:
    fig_tree = px.treemap(
        tree_data,
        path=["strategy", "symbol"],
        values="abs_return",
        color="sharpe_ratio",
        color_continuous_scale="RdYlGn",
        title="Strategy-Symbol Breakdown (size = |return|, color = Sharpe)",
        hover_data={"total_return": ":.2%", "win_rate": ":.1%"},
    )
    fig_tree.update_layout(height=600)
    fig_tree.update_traces(
        textinfo="label+value",
        texttemplate="%{label}<br>Sharpe: %{color:.2f}",
    )
    st.plotly_chart(fig_tree, use_container_width=True)

# ──────────────────────────────────────────────────
# 5. PARALLEL COORDINATES
# ──────────────────────────────────────────────────
st.header("Multi-Metric Explorer")

par_cols = ["sharpe_ratio", "total_return", "win_rate", "max_drawdown", "profit_factor", "n_trades"]
par_cols = [c for c in par_cols if c in combined.columns]

if par_cols:
    strat_map = {s: i for i, s in enumerate(combined["strategy"].unique())}
    combined_copy = combined.copy()
    combined_copy["strategy_id"] = combined_copy["strategy"].map(strat_map)

    dims = []
    for c in par_cols:
        dims.append(dict(
            range=[combined_copy[c].min(), combined_copy[c].max()],
            label=c.replace("_", " ").title(),
            values=combined_copy[c],
        ))

    fig_par = go.Figure(data=go.Parcoords(
        line=dict(
            color=combined_copy["strategy_id"],
            colorscale="Turbo",
            showscale=True,
            colorbar=dict(
                tickvals=list(strat_map.values()),
                ticktext=list(strat_map.keys()),
            ),
        ),
        dimensions=dims,
    ))
    fig_par.update_layout(title="Parallel Coordinates -- Drag axes to filter", height=500)
    st.plotly_chart(fig_par, use_container_width=True)

# ──────────────────────────────────────────────────
# DOWNLOAD BUTTON
# ──────────────────────────────────────────────────
st.download_button(
    label="Download data as CSV",
    data=combined.to_csv(index=False),
    file_name="signal_heatmap_data.csv",
    mime="text/csv",
)

# ──────────────────────────────────────────────────
# 6. DATA TABLE
# ──────────────────────────────────────────────────
with st.expander("View Raw Data", expanded=False):
    st.dataframe(
        combined.style.format({
            "net_pnl": "${:,.2f}", "total_return": "{:.2%}", "sharpe_ratio": "{:.3f}",
            "sortino_ratio": "{:.3f}", "max_drawdown": "{:.2%}", "win_rate": "{:.2%}",
            "profit_factor": "{:.3f}", "volatility": "{:.4f}",
        }),
        use_container_width=True,
    )

page_footer()
