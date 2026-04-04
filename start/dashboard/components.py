"""
Shared dashboard components.

Metric cards, filters, and reusable UI elements for Streamlit pages.
"""

import streamlit as st
import pandas as pd
from pathlib import Path


def metric_card(label: str, value, delta=None, delta_color="normal"):
    """Display a styled metric card."""
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def get_available_symbols_and_intervals(features_dir: Path) -> tuple:
    """
    Scan feature directory and return (symbols, intervals).

    Filenames are like AAPL_1h.parquet, AAPL_5min.parquet.
    Returns deduplicated symbol list and logically-ordered interval list.
    """
    import re
    if not features_dir.exists():
        return [], []

    files = list(features_dir.glob("*.parquet"))
    if not files:
        return [], []

    # Extract raw intervals from filenames
    raw_intervals = set()
    raw_symbols = set()
    for f in files:
        stem = f.stem
        # Match known interval suffixes
        m = re.match(r"^(.+?)_(5min|15min|1h|1d)$", stem)
        if m:
            raw_symbols.add(m.group(1))
            raw_intervals.add(m.group(2))
        else:
            # No interval suffix — treat entire stem as symbol
            raw_symbols.add(stem)

    # Order intervals logically: finest → coarsest
    # Always include "1d" — we can aggregate from intraday data on the fly
    if raw_intervals:  # if we have any intraday data, daily is available
        raw_intervals.add("1d")
    interval_order = ["5min", "15min", "1h", "1d"]
    intervals = [i for i in interval_order if i in raw_intervals]

    return sorted(raw_symbols), intervals


def symbol_interval_selector(features_dir: Path, default_symbol: str = "AAPL",
                              default_interval: str = "1h", layout: str = "columns"):
    """
    Reusable Symbol + Interval selector.

    layout: "columns" puts them side-by-side, "stacked" puts them vertically.
    Returns (symbol, interval).
    """
    symbols, intervals = get_available_symbols_and_intervals(features_dir)

    if not symbols:
        st.warning("No feature files found. Run the data pipeline first.")
        st.stop()

    if not intervals:
        intervals = ["1h"]

    sym_idx = symbols.index(default_symbol) if default_symbol in symbols else 0
    int_idx = intervals.index(default_interval) if default_interval in intervals else 0

    if layout == "columns":
        c1, c2 = st.columns(2)
        with c1:
            symbol = st.selectbox("Symbol", symbols, index=sym_idx)
        with c2:
            interval = st.selectbox("Interval", intervals, index=int_idx,
                                     help="5min = intraday (60-day window), 1h = hourly, 1d = daily")
    else:
        symbol = st.selectbox("Symbol", symbols, index=sym_idx)
        interval = st.selectbox("Interval", intervals, index=int_idx,
                                 help="5min = intraday (60-day window), 1h = hourly, 1d = daily")

    return symbol, interval


def auth_check():
    """
    Simple authentication gate for Streamlit Cloud.

    On Streamlit Cloud, use st.secrets for email whitelist.
    Locally, authentication is bypassed.
    """
    if not hasattr(st, "secrets"):
        return True  # Local mode, no auth

    try:
        allowed = st.secrets.get("allowed_emails", [])
        if not allowed:
            return True  # No whitelist configured

        # Streamlit Cloud provides user email via experimental_user
        if hasattr(st, "experimental_user"):
            user_email = st.experimental_user.get("email", "")
            if user_email in allowed:
                return True
            else:
                st.error("Access denied. Contact the project administrator.")
                st.stop()
                return False
    except Exception:
        return True  # Fallback: allow access


def page_footer():
    """Standard footer for all pages."""
    st.markdown("---")
    st.caption(
        "This is a research project for educational purposes only. "
        "Not financial advice. Past performance does not guarantee future results."
    )
    st.caption(
        "START — Strategic Technical Analysis for Reliable Trading | "
        "M.S. Data Analytics Capstone | McDaniel College"
    )
