"""Sentiment Analysis — LLM-based sentiment scoring results."""

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

st.set_page_config(page_title="Sentiment", page_icon="📰", layout="wide")
st.title("📰 Sentiment Analysis")
st.markdown("""
> **What this page shows:** We used an AI language model (Mistral 7B) to read financial news headlines
> and score them as *positive* (bullish), *negative* (bearish), or *neutral*. This page shows what the AI found.
""")

root = get_project_root()

try:
    scores = load_results("sentiment_scores")
    if scores.empty:
        raise FileNotFoundError
except Exception:
    st.warning("No sentiment results found. Run `python scripts/05_run_sentiment.py` first.")
    st.stop()

# ──────────────────────────────────────────────────
# 1. SENTIMENT GAUGE / OVERVIEW
# ──────────────────────────────────────────────────
st.header("Overall Market Sentiment")
st.caption("The gauge shows the average sentiment across all 12 symbols. -1 = extremely bearish, +1 = extremely bullish, 0 = neutral.")

avg_sentiment = scores["weighted_sentiment"].mean() if "weighted_sentiment" in scores.columns else 0

fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=avg_sentiment,
    domain={"x": [0, 1], "y": [0, 1]},
    title={"text": "Aggregate Market Sentiment", "font": {"size": 20}},
    delta={"reference": 0, "position": "top"},
    gauge={
        "axis": {"range": [-1, 1], "tickvals": [-1, -0.5, 0, 0.5, 1],
                 "ticktext": ["Very Bearish", "Bearish", "Neutral", "Bullish", "Very Bullish"]},
        "bar": {"color": "#2196F3"},
        "bgcolor": "white",
        "steps": [
            {"range": [-1, -0.3], "color": "#ffcdd2"},
            {"range": [-0.3, 0.3], "color": "#f5f5f5"},
            {"range": [0.3, 1], "color": "#c8e6c9"},
        ],
        "threshold": {
            "line": {"color": "black", "width": 3},
            "thickness": 0.8,
            "value": avg_sentiment,
        },
    },
))
fig_gauge.update_layout(height=300)
st.plotly_chart(fig_gauge, use_container_width=True)

# ──────────────────────────────────────────────────
# 2. SENTIMENT BY SYMBOL — LOLLIPOP CHART
# ──────────────────────────────────────────────────
st.header("Sentiment by Symbol")
st.caption("Each dot shows a symbol's sentiment score. Dots to the right are more bullish; dots to the left are more bearish.")

if "symbol" in scores.columns and "weighted_sentiment" in scores.columns:
    sorted_scores = scores.sort_values("weighted_sentiment")

    fig_lollipop = go.Figure()

    # Stems
    for _, row in sorted_scores.iterrows():
        color = "#4CAF50" if row["weighted_sentiment"] > 0 else "#F44336" if row["weighted_sentiment"] < 0 else "#9E9E9E"
        fig_lollipop.add_trace(go.Scatter(
            x=[0, row["weighted_sentiment"]],
            y=[row["symbol"], row["symbol"]],
            mode="lines",
            line=dict(color=color, width=3),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Dots
    colors = ["#4CAF50" if v > 0 else "#F44336" if v < 0 else "#9E9E9E"
              for v in sorted_scores["weighted_sentiment"]]
    fig_lollipop.add_trace(go.Scatter(
        x=sorted_scores["weighted_sentiment"],
        y=sorted_scores["symbol"],
        mode="markers",
        marker=dict(size=16, color=colors, line=dict(width=2, color="white")),
        showlegend=False,
        hovertemplate="Symbol: %{y}<br>Sentiment: %{x:.3f}<extra></extra>",
    ))

    fig_lollipop.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_lollipop.update_layout(
        title="Weighted Sentiment Score by Symbol",
        xaxis_title="Sentiment Score (-1 = Bearish, +1 = Bullish)",
        height=450,
        xaxis=dict(range=[-0.15, 0.15]),
    )
    st.plotly_chart(fig_lollipop, use_container_width=True)

# ──────────────────────────────────────────────────
# 3. SENTIMENT COMPOSITION — STACKED BAR
# ──────────────────────────────────────────────────
st.header("Sentiment Composition")
st.caption("For each symbol, what percentage of headlines were positive, negative, or neutral?")

if all(c in scores.columns for c in ["positive_pct", "negative_pct", "neutral_pct", "symbol"]):
    fig_stack = go.Figure()

    fig_stack.add_trace(go.Bar(
        name="Positive",
        x=scores["symbol"],
        y=scores["positive_pct"] * 100,
        marker_color="#4CAF50",
        hovertemplate="Positive: %{y:.1f}%<extra></extra>",
    ))
    fig_stack.add_trace(go.Bar(
        name="Neutral",
        x=scores["symbol"],
        y=scores["neutral_pct"] * 100,
        marker_color="#BDBDBD",
        hovertemplate="Neutral: %{y:.1f}%<extra></extra>",
    ))
    fig_stack.add_trace(go.Bar(
        name="Negative",
        x=scores["symbol"],
        y=scores["negative_pct"] * 100,
        marker_color="#F44336",
        hovertemplate="Negative: %{y:.1f}%<extra></extra>",
    ))

    fig_stack.update_layout(
        barmode="stack",
        title="Headline Sentiment Breakdown by Symbol",
        yaxis_title="Percentage (%)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

# ──────────────────────────────────────────────────
# 4. PIE + CONFIDENCE
# ──────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Overall Distribution")
    st.caption("Aggregate sentiment across all symbols.")
    if all(c in scores.columns for c in ["positive_pct", "negative_pct", "neutral_pct"]):
        avg_dist = scores[["positive_pct", "negative_pct", "neutral_pct"]].mean()
        fig_pie = go.Figure(data=[go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=avg_dist.values,
            marker=dict(colors=["#4CAF50", "#F44336", "#9E9E9E"]),
            hole=0.45,
            textinfo="label+percent",
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )])
        fig_pie.update_layout(
            height=380,
            annotations=[dict(text="Sentiment", x=0.5, y=0.5, font_size=14, showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    st.subheader("LLM Confidence Scores")
    st.caption("How confident was the AI in its sentiment ratings? Higher = more certain.")
    if "mean_confidence" in scores.columns:
        fig_conf = go.Figure()
        fig_conf.add_trace(go.Bar(
            x=scores["symbol"],
            y=scores["mean_confidence"],
            marker=dict(
                color=scores["mean_confidence"],
                colorscale="Viridis",
                line=dict(width=1, color="white"),
            ),
            hovertemplate="Symbol: %{x}<br>Confidence: %{y:.3f}<extra></extra>",
        ))
        fig_conf.update_layout(
            title="Mean LLM Confidence by Symbol",
            yaxis_title="Confidence (0-1)",
            height=380,
        )
        st.plotly_chart(fig_conf, use_container_width=True)

# ──────────────────────────────────────────────────
# 5. HEADLINE EXPLORER
# ──────────────────────────────────────────────────
st.header("Headline Explorer")
st.caption("Browse the actual headlines analyzed by the AI. Each headline was scored for sentiment polarity and confidence.")

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

            # Color-coded sentiment
            if "sentiment" in hl_df.columns:
                sent_filter = st.radio("Filter", ["All", "Positive", "Negative", "Neutral"], horizontal=True)
                if sent_filter == "Positive":
                    hl_df = hl_df[hl_df["sentiment"] > 0]
                elif sent_filter == "Negative":
                    hl_df = hl_df[hl_df["sentiment"] < 0]
                elif sent_filter == "Neutral":
                    hl_df = hl_df[hl_df["sentiment"] == 0]

            st.dataframe(hl_df, use_container_width=True, height=400)
    else:
        st.info("No headline files found.")
else:
    st.info("Sentiment directory not found.")

# ──────────────────────────────────────────────────
# 6. KEY INSIGHT BOX
# ──────────────────────────────────────────────────
st.header("Key Insight")

if "symbol" in scores.columns and "weighted_sentiment" in scores.columns:
    most_bullish = scores.loc[scores["weighted_sentiment"].idxmax()]
    most_bearish = scores.loc[scores["weighted_sentiment"].idxmin()]
    st.info(f"""
**Finding:** Live sentiment from Alpha Vantage news feeds produces **differentiated scores** across symbols,
unlike generic PhraseBank data which showed near-zero scores for all tickers. Real headlines tied to specific
companies yield meaningful bullish/bearish separation.

**Strongest Bullish:** `{most_bullish['symbol']}` (weighted sentiment {most_bullish['weighted_sentiment']:.3f})
**Strongest Bearish:** `{most_bearish['symbol']}` (weighted sentiment {most_bearish['weighted_sentiment']:.3f})

**Implication:** Symbol-specific live news sentiment provides actionable signal differentiation
that generic financial phrase datasets cannot. This validates the use of real-time news APIs
as a feature source for the trading framework.
""")
else:
    st.info("Sentiment scores loaded but missing expected columns for insight summary.")

# ──────────────────────────────────────────────────
# DOWNLOAD BUTTON
# ──────────────────────────────────────────────────
st.download_button(
    label="Download sentiment scores as CSV",
    data=scores.to_csv(index=False),
    file_name="sentiment_scores.csv",
    mime="text/csv",
)

page_footer()
