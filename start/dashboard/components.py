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


def symbol_selector(available: list, default: str = "AAPL") -> str:
    """Reusable symbol selector in sidebar."""
    idx = available.index(default) if default in available else 0
    return st.sidebar.selectbox("Symbol", available, index=idx)


def interval_selector() -> str:
    """Reusable interval selector in sidebar."""
    return st.sidebar.selectbox("Interval", ["1h", "1d"], index=0)


def get_available_symbols(features_dir: Path) -> list:
    """Get list of symbols with feature files."""
    if not features_dir.exists():
        return []
    return sorted([
        f.stem.replace("_1h", "").replace("_1d", "")
        for f in features_dir.glob("*.parquet")
    ])


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
        "START — Strategic Technical Analysis for Reliable Trading | "
        "M.S. Data Analytics Capstone | McDaniel College"
    )
