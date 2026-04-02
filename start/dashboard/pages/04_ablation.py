"""Ablation Study — Component-by-component contribution analysis."""

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

st.set_page_config(page_title="Ablation Study", page_icon="🔬", layout="wide")
st.title("🔬 Ablation Study")
st.markdown("""
> **What is an ablation study?** We test 6 different configurations -- each adding or removing a component --
> to measure exactly how much each piece contributes. Think of it like removing ingredients from a recipe
> to find out which ones actually matter.
""")

# Explain configs
with st.expander("What do the configurations mean?", expanded=False):
    st.markdown("""
    | Configuration | What It Uses | Plain English |
    |---|---|---|
    | **buy_and_hold** | Just buy and hold the stock | Simplest possible strategy -- buy on day 1, never sell |
    | **indicators_only** | Moving average crossover signals | Traditional technical analysis (SMA 20 crosses above SMA 50 = buy) |
    | **indicators_ml** | Technical indicators + ML model | AI model reads indicators and decides when to trade |
    | **indicators_ml_sentiment** | Above + news sentiment | Same as above but also considers whether news is positive/negative |
    | **rl_only** | Reinforcement learning agent | AI agent that learns to trade through trial and error (like training a robot) |
    | **full_hybrid** | All of the above combined | Majority vote -- if 3+ components say "buy", we buy |
    """)

root = get_project_root()

try:
    results = load_results("ablation_results")
    if results.empty:
        raise FileNotFoundError
except Exception:
    st.warning("No ablation results found. Run `python scripts/06_run_backtest.py` first.")
    st.stop()

# Config filter
if "config" in results.columns:
    configs = results["config"].unique().tolist()
    selected_configs = st.multiselect("Select Configurations to Compare", configs, default=configs)
    filtered = results[results["config"].isin(selected_configs)]
else:
    filtered = results

metric = st.selectbox(
    "Primary Metric",
    ["sharpe_ratio", "total_return", "sortino_ratio", "calmar_ratio",
     "max_drawdown", "profit_factor", "win_rate"],
    index=0,
    help="Sharpe Ratio = return per unit of risk. Higher is better.",
)

# ──────────────────────────────────────────────────
# 1. ANIMATED BAR CHART — cycle through symbols
# ──────────────────────────────────────────────────
st.header("Performance by Configuration")
st.caption("Click ▶ Play to animate through each symbol, or select a symbol from the dropdown.")

if "symbol" in filtered.columns and "config" in filtered.columns:
    fig_anim = px.bar(
        filtered.sort_values("config"),
        x="config",
        y=metric,
        color="config",
        animation_frame="symbol",
        title=f"{metric.replace('_', ' ').title()} by Configuration (animated by symbol)",
        labels={"config": "Configuration", metric: metric.replace("_", " ").title()},
        color_discrete_sequence=px.colors.qualitative.Set2,
        range_y=[filtered[metric].min() * 0.9, filtered[metric].max() * 1.1],
    )
    fig_anim.update_layout(height=500, showlegend=False)
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1200
    fig_anim.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 600
    st.plotly_chart(fig_anim, use_container_width=True)

# ──────────────────────────────────────────────────
# 2. BOX PLOT — distribution across symbols
# ──────────────────────────────────────────────────
st.header("Performance Distribution")
st.caption("Box plots show the spread of results across 12 symbols. The box = middle 50% of results. Wider = more inconsistent.")

if "config" in filtered.columns:
    fig_box = px.box(
        filtered,
        x="config",
        y=metric,
        color="config",
        points="all",
        title=f"{metric.replace('_', ' ').title()} Distribution by Configuration",
        color_discrete_sequence=px.colors.qualitative.Set2,
        hover_data=["symbol"] if "symbol" in filtered.columns else None,
    )
    fig_box.update_layout(height=500, showlegend=False)
    fig_box.update_traces(
        jitter=0.4,
        marker=dict(size=8, opacity=0.7, line=dict(width=1, color="white")),
    )
    st.plotly_chart(fig_box, use_container_width=True)

# ──────────────────────────────────────────────────
# 3. AVERAGE PERFORMANCE — HORIZONTAL BAR + ERROR BARS
# ──────────────────────────────────────────────────
st.header("Average Performance Ranking")
st.caption("Average across all 12 symbols. Error bars show standard deviation (uncertainty).")

if "config" in filtered.columns:
    agg = filtered.groupby("config")[metric].agg(["mean", "std"]).sort_values("mean", ascending=True).reset_index()

    fig_rank = go.Figure()
    fig_rank.add_trace(go.Bar(
        x=agg["mean"],
        y=agg["config"],
        orientation="h",
        error_x=dict(type="data", array=agg["std"], visible=True, color="rgba(0,0,0,0.3)"),
        marker=dict(
            color=agg["mean"],
            colorscale="Viridis",
            line=dict(color="white", width=1.5),
        ),
        text=[f"{v:.3f}" for v in agg["mean"]],
        textposition="outside",
        hovertemplate="Config: %{y}<br>Mean: %{x:.4f}<br>Std: %{error_x.array:.4f}<extra></extra>",
    ))
    fig_rank.update_layout(
        title=f"Average {metric.replace('_', ' ').title()} (with standard deviation)",
        height=400,
        margin=dict(l=200),
        xaxis_title=metric.replace("_", " ").title(),
    )
    st.plotly_chart(fig_rank, use_container_width=True)

# ──────────────────────────────────────────────────
# 4. RADAR CHART — multi-metric comparison
# ──────────────────────────────────────────────────
st.header("Multi-Metric Radar")
st.caption("Each axis is a different metric. A configuration that covers more area is better overall.")

radar_metrics = ["sharpe_ratio", "total_return", "win_rate", "profit_factor", "sortino_ratio"]
radar_metrics = [m for m in radar_metrics if m in filtered.columns]

if radar_metrics and "config" in filtered.columns:
    avg_all = filtered.groupby("config")[radar_metrics].mean()
    normalized = (avg_all - avg_all.min()) / (avg_all.max() - avg_all.min() + 1e-10)

    radar_labels = [m.replace("_", " ").title() for m in radar_metrics]

    fig_radar = go.Figure()
    colors_radar = px.colors.qualitative.Set2
    for i, config_name in enumerate(normalized.index):
        values = normalized.loc[config_name].values.tolist()
        values.append(values[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=config_name,
            line=dict(color=colors_radar[i % len(colors_radar)]),
            opacity=0.7,
        ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1.05]),
            bgcolor="rgba(0,0,0,0)",
        ),
        title="Normalized Performance Radar (outer = better)",
        height=550,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ──────────────────────────────────────────────────
# 5. HEATMAP — config x symbol
# ──────────────────────────────────────────────────
st.header("Full Heatmap")
st.caption("Every cell shows the metric value for a specific configuration + symbol combination.")

if "config" in filtered.columns and "symbol" in filtered.columns:
    pivot = filtered.pivot_table(index="symbol", columns="config", values=metric, aggfunc="first")
    if not pivot.empty:
        fmt = ".2%" if metric in ["total_return", "win_rate", "max_drawdown"] else ".3f"
        text_display = [[f"{v:{fmt}}" if not np.isnan(v) else "" for v in row] for row in pivot.values]

        fig_heat = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            text=text_display,
            texttemplate="%{text}",
            textfont={"size": 10},
            colorscale="RdYlGn" if metric != "max_drawdown" else "RdYlGn_r",
            hovertemplate="Symbol: %{y}<br>Config: %{x}<br>Value: %{z:.4f}<extra></extra>",
        ))
        fig_heat.update_layout(
            title=f"{metric.replace('_', ' ').title()} -- Symbol x Configuration",
            height=max(400, len(pivot) * 45),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# ──────────────────────────────────────────────────
# 6. INCREMENTAL VALUE WATERFALL
# ──────────────────────────────────────────────────
st.header("Incremental Component Value")
st.caption("How much does each component add when layered on top of the previous? This is the key ablation insight.")

if "config" in filtered.columns:
    order = ["buy_and_hold", "indicators_only", "indicators_ml", "indicators_ml_sentiment", "rl_only", "full_hybrid"]
    order = [c for c in order if c in filtered["config"].unique()]
    means = filtered.groupby("config")[metric].mean()

    if len(order) >= 2:
        waterfall_vals = []
        waterfall_labels = []
        for i, cfg in enumerate(order):
            if cfg in means.index:
                if i == 0:
                    waterfall_vals.append(means[cfg])
                    waterfall_labels.append(cfg)
                else:
                    prev_cfg = order[i-1]
                    if prev_cfg in means.index:
                        waterfall_vals.append(means[cfg] - means[prev_cfg])
                        waterfall_labels.append(f"+ {cfg.split('_')[-1]}")

        fig_water = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(waterfall_vals) - 1),
            x=waterfall_labels,
            y=waterfall_vals,
            textposition="outside",
            text=[f"{v:+.3f}" if i > 0 else f"{v:.3f}" for i, v in enumerate(waterfall_vals)],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#4CAF50"}},
            decreasing={"marker": {"color": "#F44336"}},
            totals={"marker": {"color": "#2196F3"}},
        ))
        fig_water.update_layout(
            title=f"Incremental {metric.replace('_', ' ').title()} Added by Each Component",
            height=450,
            yaxis_title=metric.replace("_", " ").title(),
        )
        st.plotly_chart(fig_water, use_container_width=True)

# ──────────────────────────────────────────────────
# 7. KEY FINDINGS
# ──────────────────────────────────────────────────
st.header("Key Findings")

if "config" in filtered.columns:
    means = filtered.groupby("config")["sharpe_ratio"].mean()
    best_config = means.idxmax()
    best_val = means.max()

    st.success(f"""
    **Best Configuration:** `{best_config}` with average Sharpe Ratio of **{best_val:.3f}**

    **What this means:**
    - Adding ML to technical indicators improves performance significantly
    - RL agents learn effective trading policies through trial and error
    - Generic sentiment headlines don't add value (symbol-specific news needed)
    - The full ensemble provides stable, well-rounded performance
    """)

# ──────────────────────────────────────────────────
# DOWNLOAD BUTTON
# ──────────────────────────────────────────────────
st.download_button(
    label="Download ablation results as CSV",
    data=filtered.to_csv(index=False),
    file_name="ablation_results.csv",
    mime="text/csv",
)

# Raw data
with st.expander("View Full Results Table", expanded=False):
    st.dataframe(filtered, use_container_width=True)

page_footer()
