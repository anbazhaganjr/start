"""Sentiment Analysis — LLM-based sentiment scoring results."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import get_project_root
from start.data.storage import load_results

st.set_page_config(page_title="Sentiment", page_icon="📰", layout="wide")
st.title("📰 Sentiment Analysis")

root = get_project_root()

# Load aggregate scores
try:
    scores = load_results("sentiment_scores")
    if scores.empty:
        raise FileNotFoundError
except Exception:
    st.warning("No sentiment results found. Run `python scripts/05_run_sentiment.py` first.")
    st.info("The sentiment pipeline uses Financial PhraseBank as a fallback when Ollama is not running.")
    st.stop()

# Overview
st.subheader("Sentiment Scores by Symbol")

if "symbol" in scores.columns and "weighted_sentiment" in scores.columns:
    fig = px.bar(
        scores,
        x="symbol",
        y="weighted_sentiment",
        color="weighted_sentiment",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title="Weighted Sentiment Score by Symbol",
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Sentiment distribution
col1, col2 = st.columns(2)

with col1:
    if all(c in scores.columns for c in ["positive_pct", "negative_pct", "neutral_pct"]):
        avg_dist = scores[["positive_pct", "negative_pct", "neutral_pct"]].mean()
        fig2 = px.pie(
            values=avg_dist.values,
            names=["Positive", "Negative", "Neutral"],
            title="Average Sentiment Distribution",
            color_discrete_sequence=["#4CAF50", "#F44336", "#9E9E9E"],
        )
        st.plotly_chart(fig2, use_container_width=True)

with col2:
    st.subheader("Summary Statistics")
    st.dataframe(
        scores.describe().style.format("{:.3f}"),
        use_container_width=True,
    )

# Per-symbol headlines
st.subheader("Headlines by Symbol")
sentiment_dir = root / "data" / "sentiment"

if sentiment_dir.exists():
    headline_files = sorted(sentiment_dir.glob("*_headlines.parquet"))
    if headline_files:
        symbol_select = st.selectbox(
            "Select Symbol",
            [f.stem.replace("_headlines", "") for f in headline_files],
        )
        hl_path = sentiment_dir / f"{symbol_select}_headlines.parquet"
        if hl_path.exists():
            hl_df = pd.read_parquet(hl_path)
            st.dataframe(hl_df, use_container_width=True)
    else:
        st.info("No headline files found.")
else:
    st.info("Sentiment directory not found.")
