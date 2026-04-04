# START — Strategic Technical Analysis for Reliable Trading

**M.S. Data Analytics Capstone Project | McDaniel College**

An adaptive multi-interval trading framework combining technical indicators, machine learning, reinforcement learning, and LLM-based sentiment analysis with cost-aware backtesting, ablation studies, live market data, and real-time signal generation.

**Live Dashboard**: Hosted on [Streamlit Community Cloud](https://share.streamlit.io) (access by invitation)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment variables
cp .env.example .env  # Edit with your API keys (optional — yfinance works without keys)

# 3. Run the full pipeline (one command)
python run_pipeline.py

# 4. Launch the dashboard
streamlit run start/dashboard/app.py
```

**Quick mode** (fewer epochs, faster):
```bash
python run_pipeline.py --quick
```

**Skip specific steps:**
```bash
python run_pipeline.py --skip-sentiment --skip-rl
```

---

## Architecture

```
Data Ingestion → Feature Engineering → Model Training → Backtesting → Dashboard
  (yfinance +      (17 TA indicators)    (ML + RL)      (Cost-aware)   (Streamlit)
   Tradier live)                              ↓
                                       Sentiment (LLM)
                                              ↓
                                      Ablation Studies
                                              ↓
                                     Live Signal Engine
```

### Symbol Universe (Ring 1 — 12 Symbols)
SPY, QQQ, NVDA, AAPL, TSLA, MSFT, AMZN, META, GOOGL, AMD, NFLX, AVGO

### Multi-Interval Analysis
| Interval | Source | Window | Use Case |
|----------|--------|--------|----------|
| 5-minute | yfinance | 60 days | Intraday scalping patterns |
| 1-hour | yfinance | 2 years | Primary analysis timeframe |
| Daily | Aggregated from 1h | 2 years | Swing trading, regime analysis |

---

## Project Structure

```
capstone/
├── run_pipeline.py              # Single orchestrator — runs everything
├── config/
│   ├── __init__.py              # Config loader (settings.yaml + .env)
│   └── settings.yaml            # 85+ tunable parameters (no hardcoded values)
├── start/                       # Main Python package
│   ├── data/                    # Ingestion, cleaning, Parquet storage, Tradier live quotes
│   ├── features/                # Technical indicators, returns, feature builder
│   ├── models/                  # Classical ML, LSTM, CNN, baselines, live signals
│   ├── rl/                      # Gymnasium env, PPO, DQN, reward shaping
│   ├── sentiment/               # Ollama client, Alpha Vantage news, scorer
│   ├── backtest/                # Engine, metrics, regime analysis, ablation
│   ├── dashboard/               # Streamlit app + 7 interactive pages
│   └── utils/                   # Logger, constants
├── scripts/                     # Pipeline steps (numbered for execution order)
│   ├── 01_data_ingestion.py
│   ├── 02_feature_engineering.py
│   ├── 03_model_training.py
│   ├── 04_reinforcement_learning.py
│   ├── 05_sentiment_analysis.py
│   ├── 06_ablation_backtest.py
│   └── 07_dashboard_launcher.py
├── tests/                       # Unit tests (pytest)
├── data/                        # Generated data (gitignored except results)
├── docs/                        # Academic documents + research paper
└── legacy/                      # Reference notebooks from ANA-522
```

---

## Components

### 1. Data Pipeline
- **Multi-source ingestion**: yfinance (primary, historical), Tradier (live quotes), Alpaca (alternate historical)
- **Multi-interval**: 5min, 1h natively; daily aggregated on-the-fly (Open=first, High=max, Low=min, Close=last, Volume=sum)
- **Cleaning**: Session alignment, gap detection, outlier flagging, OHLCV validation
- **Storage**: Parquet format, partitioned by symbol and interval
- **Live data**: Tradier API provides real-time quotes for all 12 symbols in a single batch call

### 2. Feature Engineering (17 Features)
- **Technical indicators**: MACD(12,26,9), RSI(14), Bollinger Bands(20,2), SMA(20/50), VWAP distance
- **Returns**: Simple, log, rolling volatility (5-bar), cumulative, intraday range, volume ratio
- **Causal design**: Target is next-bar direction — no look-ahead bias
- **Correlation filter**: Drops features with |r| > 0.95

### 3. Machine Learning Models
| Model | Type | Key Params |
|-------|------|------------|
| Logistic Regression | Classification | C=1.0, StandardScaler |
| Ridge Regression | Regression to Binary | alpha=1.0 |
| Random Forest | Classification | n_estimators=50, max_depth=8 |
| LSTM | Deep Learning | hidden=32, 1 layer, seq_len=20 |
| CNN | Deep Learning | 16 to 8 filters, k=5/3 |

**Validation**: Walk-forward expanding/sliding windows (never k-fold for time series)

### 4. Reinforcement Learning
| Agent | Algorithm | Action Space |
|-------|-----------|-------------|
| PPO | Proximal Policy Optimization | Hold / Buy / Sell |
| DQN | Deep Q-Network | Hold / Buy / Sell |

**Reward**: After-cost PnL with drawdown penalty. Long-only (no shorts).

### 5. Sentiment Analysis
- **LLM**: Mistral 7B (Q4_K_M quantization) via Ollama (local inference)
- **News source**: Alpha Vantage News Sentiment API (600+ headlines across 12 symbols)
- **Fallback**: Financial PhraseBank (Malo et al., 2014) — 4,846 annotated sentences
- **Scoring**: Polarity (-1/0/+1) with confidence (0-1)
- **Integration**: Sentiment modulates ML signals in ablation studies

### 6. Cost-Aware Backtesting
- **Slippage**: 0.01% per trade (Kissell, 2013)
- **Commission**: $0.005/share
- **Metrics**: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, turnover
- **Regime analysis**: SPY-based bull/bear/sideways classification (20-day/50-day SMA crossover)

### 7. Ablation Study (5 Configurations)
| Config | Components |
|--------|-----------|
| indicators_only | MA crossover baseline |
| indicators_ml | Technical + best ML model |
| indicators_ml_sentiment | Technical + ML + LLM sentiment |
| rl_only | PPO agent signals |
| full_hybrid | Ensemble (majority vote of all) |

### 8. Live Signal Engine
Real-time signal generation combining 8 strategies with confidence-weighted consensus voting:
- **5 rule-based**: Buy & Hold, MA Crossover, RSI Mean Reversion, MACD Momentum, Bollinger Band Position
- **3 ML models**: Logistic Regression, Ridge Regression, Random Forest (retrained on latest data)
- **Output**: BUY / SELL / HOLD with confidence score and plain-English reasoning

---

## Dashboard

7 interactive pages hosted on Streamlit Community Cloud:

| Page | Description |
|------|-------------|
| **Home** | Live ticker strip (12 symbols via Tradier API), project overview, pipeline status |
| **Signal Heatmap** | Strategy agreement across all symbols — heatmaps, bubble charts, rankings |
| **PnL Charts** | Equity curves, drawdowns, interactive candlestick charts with multi-interval support |
| **Sentiment** | LLM scoring results, sentiment gauges, headline analysis |
| **Ablation** | Component contribution analysis — animated charts, radar plots |
| **Paper Trade** | Interactive strategy simulator — pick a stock + strategy, see every trade |
| **Decisions** | Plain-English summary with live signals, market regime, risk quiz, investment calculator |

---

## Configuration

All 85+ parameters are centralized in `config/settings.yaml`:

```yaml
models:
  walk_forward:
    train_bars: 1500
    test_bars: 200
  lstm:
    hidden_size: 32
    seq_len: 20
    max_epochs: 30
rl:
  total_timesteps: 50000
  ppo:
    learning_rate: 0.0003
backtest:
  slippage_pct: 0.0001
  commission_per_share: 0.005
  initial_capital: 100000
```

API keys go in `.env` (gitignored):
```
ALPACA_API_KEY=your_key
ALPACA_API_SECRET=your_secret
TRADIER_API_KEY=your_token
ALPHA_VANTAGE_KEY=your_key
```

---

## Testing

```bash
pytest tests/ -v
```

| Test Suite | Tests | Coverage |
|-----------|-------|---------|
| test_clean.py | 4 | Data cleaning and validation |
| test_features.py | 5 | Feature engineering and leakage prevention |
| test_backtest.py | 7 | Backtesting engine and metrics |
| test_env.py | 10 | RL environment and reward functions |

---

## Cloud Deployment

The dashboard auto-deploys to **Streamlit Community Cloud** ($0 cost):

1. Push repo to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set `start/dashboard/app.py` as the main file
4. Add API keys and email whitelist in Streamlit Cloud Secrets
5. Share access by adding viewer emails in Streamlit Cloud settings

**GitHub Actions** refreshes data daily (weekdays at 6 AM EST).

---

## Memory Budget (8GB M1 Mac)

| Step | ~RAM | Notes |
|------|------|-------|
| Data/Features | ~4GB | Safe to run anytime |
| ML Training | ~4.5GB | Stop Ollama first |
| RL Training | ~4.3GB | Stop Ollama first |
| Sentiment (Ollama) | ~6.5GB | Run standalone |
| Dashboard | ~3.5GB | Reads cached results only |

---

## Key Design Decisions

1. **Walk-forward validation** — Never k-fold for time series (prevents leakage)
2. **Causal features** — Target uses future close; features use only past data
3. **CPU-only PyTorch** — Models <10K params; MPS adds overhead for zero speedup
4. **Pre-computed dashboard** — Never runs models live (prevents OOM)
5. **Parquet storage** — Columnar, compressed, fast reads
6. **Multi-source data** — Provider abstraction; adding APIs = one class + .env update
7. **Multi-interval architecture** — Daily data aggregated on-the-fly from hourly bars, no extra API calls
8. **Live signals** — Confidence-weighted consensus voting across 8 strategies
9. **Native Streamlit components** — No raw HTML; ensures compatibility with Streamlit Cloud

---

## Limitations

1. **Walk-forward accuracy ~50-52%** — Expected for next-bar prediction on financial data (Fama, 1970)
2. **RL agents overfit in-sample** — Eval returns are lower than training returns
3. **Sentiment coverage** — Alpha Vantage free tier limits headline volume
4. **Long-only** — No short selling or margin modeling
5. **Fixed position sizing** — 100 shares per trade, no dynamic allocation
6. **Live signals are indicative** — Not intended as financial advice

---

## References

- Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance, 25*(2), 383-417.
- Malo, P., et al. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology, 65*(4), 782-796.
- Schulman, J., et al. (2017). Proximal policy optimization algorithms. *arXiv:1707.06347*.
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature, 518*(7540), 529-533.
- Park, C.-H., & Irwin, S. H. (2007). What do we know about the profitability of technical analysis? *Journal of Economic Surveys, 21*(4), 786-826.
- Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management*. Academic Press.
- De Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
- Bailey, D. H., et al. (2014). The deflated Sharpe ratio. *The Journal of Portfolio Management, 40*(5), 94-107.
- Sharpe, W. F. (1994). The Sharpe ratio. *The Journal of Portfolio Management, 21*(1), 49-58.

---

## Author

**Naresh Anbazhagan** — M.S. Data Analytics, McDaniel College

---

*Built with Python, PyTorch, stable-baselines3, scikit-learn, Streamlit, Ollama, yfinance, and Tradier API.*
