"""Ablation Study — Component-by-component contribution analysis."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import get_project_root
from start.data.storage import load_results

st.set_page_config(page_title="Ablation Study", page_icon="🔬", layout="wide")
st.title("🔬 Ablation Study")
st.markdown("Isolating each component's contribution to overall performance.")

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
    selected_configs = st.multiselect("Configurations", configs, default=configs)
    filtered = results[results["config"].isin(selected_configs)]
else:
    filtered = results

# Main comparison chart
metric = st.selectbox(
    "Primary Metric",
    ["sharpe_ratio", "total_return", "sortino_ratio", "max_drawdown", "profit_factor"],
    index=0,
)

if "config" in filtered.columns and metric in filtered.columns:
    # Grouped bar chart: config × symbol
    if "symbol" in filtered.columns:
        fig = px.bar(
            filtered,
            x="config",
            y=metric,
            color="symbol",
            barmode="group",
            title=f"{metric} by Ablation Configuration",
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

    # Average across symbols
    st.subheader("Average Performance by Configuration")
    avg = filtered.groupby("config")[metric].agg(["mean", "std"]).sort_values("mean", ascending=False)
    st.dataframe(avg.style.format("{:.4f}"), use_container_width=True)

    # Radar chart for multi-metric comparison
    st.subheader("Multi-Metric Radar")
    radar_metrics = ["sharpe_ratio", "total_return", "win_rate", "profit_factor"]
    radar_metrics = [m for m in radar_metrics if m in filtered.columns]

    if radar_metrics and "config" in filtered.columns:
        avg_all = filtered.groupby("config")[radar_metrics].mean()
        # Normalize to 0-1 range for radar
        normalized = (avg_all - avg_all.min()) / (avg_all.max() - avg_all.min() + 1e-10)

        fig2 = go.Figure()
        for config_name in normalized.index:
            values = normalized.loc[config_name].values.tolist()
            values.append(values[0])  # Close the polygon
            fig2.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics + [radar_metrics[0]],
                fill="toself",
                name=config_name,
            ))
        fig2.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Normalized Performance Radar",
            height=500,
        )
        st.plotly_chart(fig2, use_container_width=True)

# Raw data
st.subheader("Full Results Table")
st.dataframe(filtered, use_container_width=True)
