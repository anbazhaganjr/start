"""Configuration loader for START project."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load .env from project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

_config = None


def get_config() -> dict:
    """Load and return project configuration, merging settings.yaml with env vars."""
    global _config
    if _config is not None:
        return _config

    settings_path = Path(__file__).parent / "settings.yaml"
    with open(settings_path, "r") as f:
        _config = yaml.safe_load(f)

    # Inject API keys from environment
    _config["api"] = {
        "tradier_key": os.environ.get("TRADIER_API_KEY", ""),
        "alpaca_key": os.environ.get("ALPACA_API_KEY", ""),
        "alpaca_secret": os.environ.get("ALPACA_API_SECRET", ""),
        "alpaca_base_url": os.environ.get(
            "ALPACA_BASE_URL", "https://paper-api.alpaca.markets"
        ),
    }

    # Resolve paths relative to project root
    for key in ("raw_dir", "parquet_dir", "features_dir"):
        _config["data"][key] = str(_PROJECT_ROOT / _config["data"][key])

    return _config


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT
