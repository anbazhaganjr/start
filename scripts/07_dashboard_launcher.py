#!/usr/bin/env python3
"""
Script 07: Launch the Streamlit dashboard.

Usage:
    python scripts/07_launch_dashboard.py
    # Or directly:
    streamlit run start/dashboard/app.py
"""

import sys
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    app_path = Path(__file__).resolve().parent.parent / "start" / "dashboard" / "app.py"
    print(f"Launching dashboard: {app_path}")
    subprocess.run(["streamlit", "run", str(app_path)], check=True)


if __name__ == "__main__":
    main()
