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
BARS_PER_DAY = 78  # 5-min bars in regular session
BAR_INTERVAL = "5Min"

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
ANNUALIZATION_FACTOR = 252 * BARS_PER_DAY  # For intraday Sharpe

# Walk-forward
WALK_FORWARD_TRAIN_DAYS = 60
WALK_FORWARD_TEST_DAYS = 5

# RL
RL_TOTAL_TIMESTEPS = 50_000
RL_DRAWDOWN_PENALTY = 0.1
