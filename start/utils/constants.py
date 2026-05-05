"""Project-wide constants."""

# Ring 1 symbols (MVP universe)
RING1_SYMBOLS = [
    "SPY", "QQQ", "NVDA", "AAPL", "TSLA",
    "MSFT", "AMZN", "META", "GOOGL", "AMD",
    "NFLX", "AVGO",
]

# Trading session
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"
BARS_PER_DAY = 78  # 5-min bars in regular session (legacy constant)
BAR_INTERVAL = "5Min"

# Bars-per-day per supported interval (regular session = 6.5 hours)
BARS_PER_DAY_BY_INTERVAL = {
    "5Min": 78,
    "5min": 78,
    "15Min": 26,
    "15min": 26,
    "1h": 7,    # 6.5 hours rounded up to whole bars (9:30, 10:30, ..., 15:30)
    "1H": 7,
    "1d": 1,
    "1D": 1,
    "daily": 1,
}

# Feature engineering
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
SMA_SHORT = 20
SMA_LONG = 50
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ROLLING_VOL_WINDOW = 5

# Backtesting defaults
SLIPPAGE_PCT = 0.0001  # 0.01%
COMMISSION_PER_SHARE = 0.005  # $0.005

# Annualization factor for intraday Sharpe / Sortino / volatility.
# Earlier versions hardcoded 252 * 78 (5-min bars), but the analysis runs on
# 1-hour bars, so the correct factor is 252 trading days * 7 hourly bars/day.
# Use get_annualization_factor(interval) below for interval-specific scaling.
ANNUALIZATION_FACTOR = 252 * BARS_PER_DAY_BY_INTERVAL["1h"]  # = 1764


def get_annualization_factor(interval: str = "1h") -> int:
    """Return Sharpe annualization factor for the given bar interval."""
    bars = BARS_PER_DAY_BY_INTERVAL.get(interval, BARS_PER_DAY_BY_INTERVAL["1h"])
    return 252 * bars

# Walk-forward
WALK_FORWARD_TRAIN_DAYS = 60
WALK_FORWARD_TEST_DAYS = 5

# RL
RL_TOTAL_TIMESTEPS = 50_000
RL_DRAWDOWN_PENALTY = 0.1
