"""
START Dashboard — Streamlit main entry point.

Loads pre-computed results from Parquet files.
Never runs models live (prevents OOM on 8GB RAM).
"""

import sys
from pathlib import Path
from datetime import datetime

import streamlit as st

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

st.set_page_config(
    page_title="START — Adaptive Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Authentication check (enforced on Streamlit Cloud)
from start.dashboard.components import auth_check, page_footer
auth_check()

from config import get_project_root, get_setting
from start.data.storage import load_results

root = get_project_root()

st.title("📈 START — Strategic Technical Analysis for Reliable Trading")
st.markdown("**M.S. Data Analytics Capstone | McDaniel College**")
st.markdown("---")

# Project Overview
st.header("Project Overview")

st.markdown("""
START is an adaptive intraday trading framework that combines:

- **Technical Analysis** — MACD, RSI, Bollinger Bands, SMA crossovers
- **Machine Learning** — Logistic Regression, Ridge, Random Forest, LSTM, CNN
- **Reinforcement Learning** — PPO and DQN agents with cost-aware rewards
- **LLM Sentiment** — Mistral 7B via Ollama for financial headline scoring
- **Cost-Aware Backtesting** — Realistic slippage and commission modeling
- **Ablation Studies** — 5-config analysis isolating each component's contribution

**Universe:** 12 high-liquidity symbols (SPY, QQQ, NVDA, AAPL, TSLA, MSFT, AMZN, META, GOOGL, AMD, NFLX, AVGO)
""")

# Key Metrics
st.header("Key Results")

col1, col2, col3, col4, col5 = st.columns(5)

# Load aggregate results
try:
    model_results = load_results("model_comparison")
    if not model_results.empty:
        best = model_results.sort_values("sharpe_ratio", ascending=False).iloc[0]
        col1.metric("Best Sharpe", f"{best['sharpe_ratio']:.3f}", f"{best.get('strategy', 'N/A')}")
        col2.metric("Symbols Tested", model_results["symbol"].nunique() if "symbol" in model_results.columns else 0)
        col3.metric("Strategies", model_results["strategy"].nunique() if "strategy" in model_results.columns else 0)
except Exception:
    col1.metric("Best Sharpe", "—")
    col2.metric("Symbols", "—")
    col3.metric("Strategies", "—")

try:
    rl_results = load_results("rl_comparison")
    if not rl_results.empty:
        best_rl = rl_results.sort_values("sharpe_ratio", ascending=False).iloc[0]
        col4.metric("Best RL Sharpe", f"{best_rl['sharpe_ratio']:.3f}", f"{best_rl.get('strategy', '')}")
except Exception:
    col4.metric("Best RL Sharpe", "—")

# Data freshness
features_dir = root / "data" / "features"
if features_dir.exists():
    latest = max(features_dir.glob("*.parquet"), key=lambda f: f.stat().st_mtime, default=None)
    if latest:
        mod_time = datetime.fromtimestamp(latest.stat().st_mtime)
        col5.metric("Data Updated", mod_time.strftime("%b %d, %Y"))
    else:
        col5.metric("Data Updated", "—")
else:
    col5.metric("Data Updated", "—")

# Architecture diagram
st.header("Architecture")
st.markdown("""
```
Data Ingestion → Feature Engineering → Model Training → Backtesting → Dashboard
    (yfinance)     (TA indicators)      (ML + RL)      (Cost-aware)   (Streamlit)
                                            ↓
                                     Sentiment (LLM)
                                            ↓
                                    Ablation Studies
```
""")

# Quick stats
st.header("Pipeline Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    n_features = len(list(features_dir.glob("*.parquet"))) if features_dir.exists() else 0
    st.metric("Feature Files", n_features)

with col2:
    results_dir = root / "data" / "results"
    n_results = len(list(results_dir.glob("*.parquet"))) if results_dir.exists() else 0
    st.metric("Result Files", n_results)

with col3:
    models_dir = root / "data" / "models"
    n_models = len(list(models_dir.glob("*"))) if models_dir.exists() else 0
    st.metric("RL Model Files", n_models)

with col4:
    sentiment_dir = root / "data" / "sentiment"
    n_sent = len(list(sentiment_dir.glob("*.parquet"))) if sentiment_dir.exists() else 0
    st.metric("Sentiment Files", n_sent)

# Navigation guide
st.header("Dashboard Pages")
st.markdown("""
| Page | Description |
|------|-------------|
| 🔥 **Signal Heatmap** | Model signal agreement across symbols and time |
| 💰 **PnL Charts** | Equity curves, drawdowns, and trade analysis |
| 📰 **Sentiment** | LLM-based sentiment scoring results |
| 🔬 **Ablation** | Component contribution analysis (5 configs) |
| 📊 **Paper Trade** | Interactive strategy simulation |

**Use the sidebar to navigate between pages.**
""")

page_footer()
