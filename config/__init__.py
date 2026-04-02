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
        "alphavantage_key": os.environ.get("ALPHAVANTAGE_API_KEY", ""),
    }

    # Resolve Alpha Vantage key in sentiment config (supports ${ENV_VAR} syntax)
    sent_cfg = _config.get("sentiment", {})
    av_key_val = sent_cfg.get("alphavantage_api_key", "")
    if isinstance(av_key_val, str) and av_key_val.startswith("${") and av_key_val.endswith("}"):
        env_name = av_key_val[2:-1]
        sent_cfg["alphavantage_api_key"] = os.environ.get(env_name, "")

    # Resolve paths relative to project root
    path_keys = ("raw_dir", "parquet_dir", "features_dir", "results_dir",
                 "models_dir", "sentiment_dir")
    for key in path_keys:
        if key in _config.get("data", {}):
            _config["data"][key] = str(_PROJECT_ROOT / _config["data"][key])

    return _config


def get_setting(*keys, default=None):
    """
    Get a nested config value using dot-style keys.

    Usage:
        get_setting("models", "lstm", "hidden_size")       → 32
        get_setting("backtest", "initial_capital")          → 100000
        get_setting("models", "lstm", "missing", default=0) → 0
    """
    cfg = get_config()
    for key in keys:
        if isinstance(cfg, dict) and key in cfg:
            cfg = cfg[key]
        else:
            return default
    return cfg


def get_project_root() -> Path:
    """Return the project root directory."""
    return _PROJECT_ROOT
