"""Signal Heatmap — Model signal agreement across symbols."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import get_project_root
from start.data.storage import load_results

st.set_page_config(page_title="Signal Heatmap", page_icon="🔥", layout="wide")
st.title("🔥 Signal Heatmap")
st.markdown("Model signal agreement across symbols and strategies.")

root = get_project_root()

try:
    results = load_results("model_comparison")
    if results.empty:
        st.warning("No model comparison results found. Run script 03 first.")
        st.stop()
except Exception:
    st.warning("No model comparison results found. Run `python scripts/03_train_models.py` first.")
    st.stop()

# Pivot: symbols × strategies
metrics_to_show = st.selectbox(
    "Metric",
    ["sharpe_ratio", "total_return", "win_rate", "max_drawdown", "n_trades"],
    index=0,
)

if "symbol" in results.columns and "strategy" in results.columns:
    pivot = results.pivot_table(
        index="symbol",
        columns="strategy",
        values=metrics_to_show,
        aggfunc="first",
    )

    if not pivot.empty:
        fig = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            labels={"color": metrics_to_show},
        )
        fig.update_layout(
            title=f"{metrics_to_show} by Symbol × Strategy",
            height=max(400, len(pivot) * 50),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Raw Data")
        st.dataframe(pivot.style.format("{:.4f}"), use_container_width=True)
else:
    st.warning("Results missing 'symbol' or 'strategy' columns.")

# Strategy comparison bar chart
st.subheader("Strategy Comparison")
if "strategy" in results.columns:
    avg_by_strategy = results.groupby("strategy")[metrics_to_show].mean().sort_values(ascending=False)
    fig2 = px.bar(
        x=avg_by_strategy.index,
        y=avg_by_strategy.values,
        labels={"x": "Strategy", "y": f"Mean {metrics_to_show}"},
        title=f"Average {metrics_to_show} by Strategy",
        color=avg_by_strategy.values,
        color_continuous_scale="RdYlGn",
    )
    st.plotly_chart(fig2, use_container_width=True)
