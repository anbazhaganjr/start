"""Decision Dashboard — The big-picture view for non-technical users."""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import get_project_root, get_config
from start.data.storage import load_results, load_features
from start.utils.constants import RING1_SYMBOLS

st.set_page_config(page_title="Decision Dashboard", page_icon="\U0001f3af", layout="wide")
st.title("\U0001f3af Decision Dashboard")
st.markdown("""
> **Welcome!** This page gives you the plain-English summary of everything our
> trading research found. No jargon, no complicated charts — just clear answers
> about what works and what doesn't.
""")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def _load_all():
    mc = load_results("model_comparison")
    rl = load_results("rl_comparison")
    # Standardise columns so we can combine them
    shared_cols = [
        "symbol", "strategy", "total_return", "sharpe_ratio", "max_drawdown",
        "annualized_return", "volatility", "win_rate", "n_trades",
        "net_pnl", "final_equity", "sortino_ratio", "calmar_ratio",
        "profit_factor",
    ]
    mc_sub = mc[[c for c in shared_cols if c in mc.columns]].copy()
    rl_sub = rl[[c for c in shared_cols if c in rl.columns]].copy()
    combined = pd.concat([mc_sub, rl_sub], ignore_index=True)
    return mc, rl, combined


mc_df, rl_df, combined_df = _load_all()

if combined_df.empty:
    st.error("No results data found. Please run the backtesting pipeline first.")
    st.stop()

STRATEGY_LABELS = {
    "buy_hold": "Buy & Hold",
    "ma_crossover": "MA Crossover",
    "logistic": "Logistic Regression",
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "rl_ppo": "AI Agent (PPO)",
    "rl_dqn": "AI Agent (DQN)",
}


def _nice(strategy: str) -> str:
    return STRATEGY_LABELS.get(strategy, strategy)


# ===================================================================
# 1. Market Regime Indicator
# ===================================================================
st.header("\U0001f6a6 Market Regime Indicator")
st.markdown("_A quick snapshot of how the overall market behaved during our test period._")

avg_return = combined_df["total_return"].mean()

if avg_return > 0.03:
    regime, color, icon = "BULL MARKET", "#27ae60", "\U0001f7e2"
    explanation = (
        "The market trended **upward** during our test window. "
        "Historically, most of our strategies perform well in this environment — "
        "even simple ones like Buy & Hold capture gains."
    )
elif avg_return < -0.01:
    regime, color, icon = "BEAR MARKET", "#e74c3c", "\U0001f534"
    explanation = (
        "The market trended **downward** during our test window. "
        "This is where smarter strategies like Ridge Regression and the AI Agent "
        "can protect your capital by stepping aside."
    )
else:
    regime, color, icon = "SIDEWAYS MARKET", "#f39c12", "\U0001f7e1"
    explanation = (
        "The market moved **sideways** — no strong trend up or down. "
        "In these conditions, the best strategies are ones that pick short-term "
        "signals rather than riding a trend."
    )

col_regime1, col_regime2 = st.columns([1, 3])

with col_regime1:
    st.subheader(f"{icon} {regime}")
    st.metric("Avg Strategy Return", f"{avg_return:.2%}")

with col_regime2:
    st.markdown(explanation)
    # Mini bar chart: average return by strategy
    strat_avg = (
        combined_df.groupby("strategy")["total_return"]
        .mean()
        .reset_index()
        .sort_values("total_return", ascending=True)
    )
    strat_avg["label"] = strat_avg["strategy"].map(_nice)
    fig = px.bar(
        strat_avg,
        x="total_return",
        y="label",
        orientation="h",
        color="total_return",
        color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
        labels={"total_return": "Average Return", "label": ""},
        title="Average Return by Strategy",
    )
    fig.update_layout(showlegend=False, coloraxis_showscale=False, height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# ===================================================================
# 2. LIVE TRADING SIGNALS (real-time from Tradier + feature analysis)
# ===================================================================
st.header("\U0001f4e1 Live Trading Signals")
st.markdown(
    "_Real-time signals based on the latest market data. "
    "Each strategy analyzes current indicators to produce a BUY, SELL, or HOLD recommendation._"
)

# Try to show live quotes
_live_quotes = {}
try:
    from start.data.providers import TradierProvider
    _cfg = get_config()
    _tradier_key = _cfg.get("api", {}).get("tradier_key", "")
    if _tradier_key:
        _tp = TradierProvider(api_token=_tradier_key)
        _qs = _tp.fetch_live_quotes(RING1_SYMBOLS)
        _live_quotes = {q["symbol"]: q for q in _qs}
except Exception:
    pass

# Generate live signals for each symbol
from start.models.live_signals import get_baseline_signals, get_signal_consensus

@st.cache_data(ttl=300)
def _compute_live_signals():
    """Compute live signals for all symbols using latest feature data."""
    results = {}
    for sym in RING1_SYMBOLS:
        try:
            df = load_features(sym, "1h")
            if df.empty:
                continue
            signals = get_baseline_signals(df)
            consensus = get_signal_consensus(signals)
            results[sym] = consensus
        except Exception:
            continue
    return results

live_signals = _compute_live_signals()

if live_signals:
    rows_of_4 = [RING1_SYMBOLS[i:i+4] for i in range(0, len(RING1_SYMBOLS), 4)]
    for row_syms in rows_of_4:
        cols = st.columns(4)
        for idx, sym in enumerate(row_syms):
            if sym not in live_signals:
                continue
            cs = live_signals[sym]
            quote = _live_quotes.get(sym, {})

            # Determine emoji based on consensus
            sig_label = cs["overall_label"]
            if sig_label == "BUY":
                traffic = "\U0001f7e2"
            elif sig_label == "SELL":
                traffic = "\U0001f534"
            else:
                traffic = "\U0001f7e1"

            with cols[idx]:
                # Use native Streamlit components — no raw HTML
                price_str = ""
                delta_str = None
                if quote:
                    price_str = f"${quote['last']:.2f}"
                    delta_str = f"{quote['change']:+.2f} ({quote['change_pct']:+.2f}%)"

                st.metric(
                    label=f"{traffic} {sym}",
                    value=price_str if price_str else sig_label,
                    delta=delta_str,
                    help=f"{cs['n_buy']} buy / {cs['n_sell']} sell / {cs['n_hold']} hold",
                )
                st.caption(f"**{sig_label}** · {cs['n_buy']}B / {cs['n_sell']}S / {cs['n_hold']}H")

    # Detailed signal breakdown for selected symbol
    st.subheader("Signal Detail")
    selected_sym = st.selectbox("Select a stock to see signal breakdown", RING1_SYMBOLS, key="signal_detail_sym")

    if selected_sym in live_signals:
        cs = live_signals[selected_sym]

        # Build a DataFrame for clean display
        detail_rows = []
        for name, sig in cs["strategies"].items():
            detail_rows.append({
                "Strategy": name,
                "Signal": f"{'🟢' if sig['signal'] == 1 else '🔴' if sig['signal'] == 0 else '🟡'} {sig['label']}",
                "Confidence": sig["confidence"],
                "Reason": sig["reason"],
            })

        detail_df = pd.DataFrame(detail_rows)
        st.dataframe(
            detail_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    format="%.0%%",
                ),
            },
        )

    st.caption("Signals refresh every 5 minutes. Based on latest hourly feature data + live prices.")

else:
    st.info("Live signals not available. Run the data pipeline to generate feature data.")

st.divider()

# ===================================================================
# 2b. Historical Signal Consensus (from backtest results)
# ===================================================================
st.header("\U0001f4ca Historical Performance Consensus")
st.markdown(
    "_Based on backtesting: for each stock, how many strategies were profitable? "
    "More green = more strategies made money historically._"
)

strategies_list = combined_df["strategy"].unique().tolist()
n_strats = len(strategies_list)

symbols = RING1_SYMBOLS
rows_of_4 = [symbols[i : i + 4] for i in range(0, len(symbols), 4)]

for row_syms in rows_of_4:
    cols = st.columns(4)
    for idx, sym in enumerate(row_syms):
        sym_data = combined_df[combined_df["symbol"] == sym]
        n_positive = (sym_data["total_return"] > 0).sum()

        if n_positive >= 4:
            traffic = "\U0001f7e2"
            hist_label = "Strong Buy"
        elif n_positive >= 2:
            traffic = "\U0001f7e1"
            hist_label = "Mixed"
        else:
            traffic = "\U0001f534"
            hist_label = "Caution"

        with cols[idx]:
            st.metric(
                label=f"{traffic} {sym}",
                value=hist_label,
                delta=f"{n_positive}/{n_strats} profitable",
                delta_color="normal" if n_positive >= 3 else ("off" if n_positive >= 2 else "inverse"),
            )

st.divider()

# ===================================================================
# 3. "What If I Invested $10,000?" Calculator
# ===================================================================
st.header("\U0001f4b0 What If I Invested $10,000?")
st.markdown("_Pick a stock and a strategy, and see what your investment would be worth today._")

calc_col1, calc_col2 = st.columns(2)
with calc_col1:
    sel_symbol = st.selectbox("Pick a stock", RING1_SYMBOLS, index=3, key="calc_sym")
with calc_col2:
    available = combined_df[combined_df["symbol"] == sel_symbol]["strategy"].unique()
    sel_strategy = st.selectbox(
        "Pick a strategy",
        available,
        format_func=_nice,
        key="calc_strat",
    )

row = combined_df[
    (combined_df["symbol"] == sel_symbol) & (combined_df["strategy"] == sel_strategy)
]

if not row.empty:
    r = row.iloc[0]
    invest = 10_000
    total_ret = r["total_return"]
    final = invest * (1 + total_ret)
    max_dd = r["max_drawdown"]
    worst_case = invest * (1 + max_dd)  # max_drawdown is negative
    # Best case: use the max return across all symbols for this strategy
    best_ret = combined_df[combined_df["strategy"] == sel_strategy]["total_return"].max()
    best_case = invest * (1 + best_ret)

    sign = "+" if total_ret >= 0 else ""

    st.markdown(
        f"If you invested **$10,000** using **{_nice(sel_strategy)}** on **{sel_symbol}**..."
    )
    st.metric(
        label="Final Portfolio Value",
        value=f"${final:,.0f}",
        delta=f"{sign}{total_ret:.2%}",
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("\U0001f4c8 Final Value", f"${final:,.0f}")
    m2.metric("\U0001f534 Worst Drawdown", f"${worst_case:,.0f}", f"{max_dd:.2%}")
    m3.metric("\U0001f7e2 Best Case (this strategy)", f"${best_case:,.0f}", f"+{best_ret:.2%}")
    m4.metric("\U0001f3af Win Rate", f"{r['win_rate']:.1%}" if pd.notna(r.get("win_rate")) else "N/A")
else:
    st.warning("No data available for this combination.")

st.divider()

# ===================================================================
# 4. Strategy Recommendation Engine
# ===================================================================
st.header("\U0001f3c6 Strategy Recommendation Engine")
st.markdown(
    "_For each stock, we found the strategy with the best risk-adjusted returns "
    "(Sharpe ratio). Higher Sharpe = better return for the risk taken._"
)

best_per_sym = (
    combined_df.sort_values("sharpe_ratio", ascending=False)
    .groupby("symbol")
    .first()
    .reset_index()
)
best_per_sym["nice_strategy"] = best_per_sym["strategy"].map(_nice)


def _recommendation(sharpe):
    if sharpe > 1.5:
        return "\U0001f7e2 Strongly Recommended"
    elif sharpe > 0.5:
        return "\U0001f7e1 Worth Considering"
    elif sharpe > 0:
        return "\U0001f7e0 Proceed With Caution"
    else:
        return "\U0001f534 Not Recommended"


best_per_sym["recommendation"] = best_per_sym["sharpe_ratio"].apply(_recommendation)

display_df = best_per_sym[["symbol", "nice_strategy", "sharpe_ratio", "total_return", "recommendation"]].copy()
display_df.columns = ["Stock", "Best Strategy", "Sharpe Ratio", "Total Return", "Recommendation"]
display_df["Sharpe Ratio"] = display_df["Sharpe Ratio"].round(3)
display_df["Total Return"] = display_df["Total Return"].apply(lambda x: f"{x:.2%}")

# Order by original symbol list
display_df["_order"] = display_df["Stock"].apply(lambda s: RING1_SYMBOLS.index(s) if s in RING1_SYMBOLS else 99)
display_df = display_df.sort_values("_order").drop(columns=["_order"])

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Sharpe Ratio": st.column_config.NumberColumn(format="%.3f"),
    },
)

# Sharpe ratio chart
fig_sharpe = px.bar(
    best_per_sym.sort_values("sharpe_ratio", ascending=True),
    x="sharpe_ratio",
    y="symbol",
    orientation="h",
    color="sharpe_ratio",
    color_continuous_scale=["#e74c3c", "#f39c12", "#27ae60"],
    text="nice_strategy",
    labels={"sharpe_ratio": "Sharpe Ratio", "symbol": ""},
    title="Best Strategy Sharpe Ratio by Stock",
)
fig_sharpe.update_layout(showlegend=False, coloraxis_showscale=False, height=420, margin=dict(l=0, r=0, t=40, b=0))
fig_sharpe.update_traces(textposition="inside")
st.plotly_chart(fig_sharpe, use_container_width=True)

st.divider()

# ===================================================================
# 5. Risk Tolerance Quiz
# ===================================================================
st.header("\U0001f9e0 What Strategy Is Right for You?")
st.markdown("_Answer three quick questions and we'll suggest a strategy that matches your comfort level._")

quiz_col1, quiz_col2, quiz_col3 = st.columns(3)

with quiz_col1:
    q1 = st.radio(
        "\U0001f4c5 How long can you hold an investment?",
        ["1 week", "1 month", "6+ months"],
        key="quiz_q1",
    )

with quiz_col2:
    q2 = st.radio(
        "\U0001f4c9 How much loss can you stomach?",
        ["5% — I panic easily", "10% — I can handle some bumps", "20%+ — I'm in it for the long haul"],
        key="quiz_q2",
    )

with quiz_col3:
    q3 = st.radio(
        "\U0001f4f1 How often do you want to check?",
        ["Daily", "Weekly", "Monthly"],
        key="quiz_q3",
    )

# Score: 0 = conservative, 1 = moderate, 2 = aggressive
score = 0
score += ["1 week", "1 month", "6+ months"].index(q1)
score += ["5% — I panic easily", "10% — I can handle some bumps", "20%+ — I'm in it for the long haul"].index(q2)
score += ["Daily", "Weekly", "Monthly"].index(q3)

if score <= 2:
    profile = "Conservative"
    rec_strategies = ["buy_hold", "ridge"]
    profile_color = "#3498db"
    profile_icon = "\U0001f6e1\ufe0f"
    profile_text = (
        "You prefer **safety and simplicity**. We recommend **Buy & Hold** or "
        "**Ridge Regression** — they trade less often, keep costs low, and don't "
        "require you to watch the market every day."
    )
elif score <= 4:
    profile = "Moderate"
    rec_strategies = ["logistic", "ridge"]
    profile_color = "#f39c12"
    profile_icon = "\u2696\ufe0f"
    profile_text = (
        "You're **comfortable with some risk** for better returns. We recommend "
        "**Logistic Regression** or **Ridge Regression** — machine learning models "
        "that adapt to changing markets while keeping risk manageable."
    )
else:
    profile = "Aggressive"
    rec_strategies = ["rl_ppo", "rl_dqn"]
    profile_color = "#e74c3c"
    profile_icon = "\U0001f680"
    profile_text = (
        "You're a **risk-taker** who wants maximum returns. We recommend the "
        "**AI Agent (PPO)** — our reinforcement learning model that earned the "
        "highest risk-adjusted returns but trades more frequently."
    )

st.subheader(f"{profile_icon} Your Profile: {profile}")
st.markdown(profile_text)

# Show performance of recommended strategies
rec_data = combined_df[combined_df["strategy"].isin(rec_strategies)]
if not rec_data.empty:
    st.subheader("How Your Recommended Strategies Performed")
    rec_summary = (
        rec_data.groupby("strategy")
        .agg({"total_return": "mean", "sharpe_ratio": "mean", "max_drawdown": "mean", "win_rate": "mean"})
        .reset_index()
    )
    rec_summary["Strategy"] = rec_summary["strategy"].map(_nice)

    rc1, rc2, rc3 = st.columns(3)
    for i, (_, srow) in enumerate(rec_summary.iterrows()):
        col = [rc1, rc2, rc3][i % 3]
        with col:
            st.markdown(f"**{srow['Strategy']}**")
            st.metric("Avg Return", f"{srow['total_return']:.2%}")
            st.metric("Avg Sharpe", f"{srow['sharpe_ratio']:.3f}")
            st.metric("Avg Max Drawdown", f"{srow['max_drawdown']:.2%}")

st.divider()

# ===================================================================
# 6. Key Takeaways
# ===================================================================
st.header("\U0001f4a1 Key Takeaways — What We Learned")
st.markdown("_The five most important lessons from our research, in plain English:_")

takeaways = [
    (
        "\U0001f4ca Simple ML models often beat complex ones",
        "Ridge Regression (a straightforward statistical model) frequently outperformed "
        "the fancier Random Forest model. Sometimes simpler is better.",
    ),
    (
        "\U0001f916 The AI trading robot earned strong risk-adjusted returns",
        "Our reinforcement learning agent (PPO) learned to trade by trial-and-error, "
        "similar to how game-playing AIs learn. It achieved some of the highest "
        "Sharpe ratios in our tests.",
    ),
    (
        "\U0001f4f0 Generic news headlines don't help trading",
        "We tested whether reading news headlines improves predictions. Surprisingly, "
        "general market news adds little value — you would need stock-specific news "
        "to gain an edge.",
    ),
    (
        "\U0001f3b2 No strategy is right 100% of the time",
        "Even our best-performing strategy only wins about 57% of its trades. "
        "The key to profit is making your wins bigger than your losses, not winning "
        "every single trade.",
    ),
    (
        "\U0001f4b8 After trading costs, patience beats frequent trading",
        "Every trade costs money (commissions and slippage). Strategies that trade "
        "less often tend to keep more of their profits. Patience pays.",
    ),
]

for title, detail in takeaways:
    with st.expander(title, expanded=True):
        st.markdown(detail)

st.divider()

# ===================================================================
# 7. Disclaimer
# ===================================================================
st.warning(
    "**Important Disclaimer:** This is a research project for educational purposes only. "
    "**Not financial advice.** Past performance does not guarantee future results. "
    "All results are from backtesting on historical data with simulated trading costs."
)
