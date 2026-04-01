# START — Strategic Technical Analysis for Reliable Trading

**M.S. Data Analytics Capstone Project | McDaniel College**

An adaptive intraday trading framework combining technical indicators, machine learning, reinforcement learning, and LLM-based sentiment analysis with cost-aware backtesting and ablation studies.

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
  (yfinance)      (TA indicators)       (ML + RL)       (Cost-aware)   (Streamlit)
                                             ↓
                                      Sentiment (LLM)
                                             ↓
                                     Ablation Studies
```

### Symbol Universe (Ring 1)
SPY, QQQ, NVDA, AAPL, TSLA, MSFT, AMZN, META, GOOGL, AMD, NFLX, AVGO

---

## Project Structure

```
capstone/
├── run_pipeline.py              # Single orchestrator — runs everything
├── config/
│   ├── __init__.py              # Config loader (settings.yaml + .env)
│   └── settings.yaml            # ALL tunable parameters (no hardcoded values)
├── start/                       # Main Python package
│   ├── data/                    # Ingestion, cleaning, storage (Parquet)
│   ├── features/                # Technical indicators, returns, feature builder
│   ├── models/                  # Classical ML, LSTM, CNN, baselines, analysis
│   ├── rl/                      # Gymnasium env, PPO, DQN, reward shaping
│   ├── sentiment/               # Ollama client, news fetcher, scorer
│   ├── backtest/                # Engine, metrics, regime analysis, ablation
│   ├── dashboard/               # Streamlit app + 5 interactive pages
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
├── docs/                        # Academic documents
└── legacy/                      # Reference notebooks from ANA-522
```

---

## Components

### 1. Data Pipeline
- **Multi-source ingestion**: yfinance (primary), Alpaca, Tradier, local files
- **Cleaning**: Session alignment, gap detection, outlier flagging, OHLCV validation
- **Storage**: Parquet format, partitioned by symbol

### 2. Feature Engineering
- **Technical indicators**: MACD(12,26,9), RSI(14), Bollinger Bands(20,2), SMA(20/50), VWAP distance
- **Returns**: Simple, log, rolling volatility (5-bar), cumulative, intraday range, volume ratio
- **Causal design**: Target is next-bar direction — no look-ahead bias
- **Correlation filter**: Drops features with |r| > 0.95

### 3. Machine Learning Models
| Model | Type | Key Params |
|-------|------|------------|
| Logistic Regression | Classification | C=1.0, StandardScaler |
| Ridge Regression | Regression→Binary | alpha=1.0 |
| Random Forest | Classification | n_estimators=50, max_depth=8 |
| LSTM | Deep Learning | hidden=32, 1 layer, seq_len=20 |
| CNN | Deep Learning | 16→8 filters, k=5/3 |

**Validation**: Walk-forward expanding/sliding windows (never k-fold for time series)

### 4. Reinforcement Learning
| Agent | Algorithm | Action Space |
|-------|-----------|-------------|
| PPO | Proximal Policy Optimization | Hold / Buy / Sell |
| DQN | Deep Q-Network | Hold / Buy / Sell |

**Reward**: After-cost PnL with drawdown penalty. Long-only (no shorts).

### 5. Sentiment Analysis
- **LLM**: Mistral 7B (Q4_K_M quantization) via Ollama
- **Fallback**: Financial PhraseBank (Malo et al., 2014)
- **Scoring**: Polarity (-1/0/+1) with confidence (0-1)
- **Integration**: Sentiment modulates ML signals in ablation

### 6. Cost-Aware Backtesting
- **Slippage**: 0.01% per trade
- **Commission**: $0.005/share
- **Metrics**: Sharpe, Sortino, Calmar, max drawdown, win rate, profit factor, turnover
- **Regime analysis**: SPY-based bull/bear/sideways classification

### 7. Ablation Study (5 Configurations)
| Config | Components |
|--------|-----------|
| indicators_only | MA crossover baseline |
| indicators_ml | Technical + best ML model |
| indicators_ml_sentiment | Technical + ML + LLM sentiment |
| rl_only | PPO agent signals |
| full_hybrid | Ensemble (majority vote of all) |

---

## Dashboard

5 interactive pages hosted on Streamlit Cloud:

1. **Signal Heatmap** — Model agreement across symbols (Plotly heatmaps)
2. **PnL Charts** — Equity curves, drawdowns, trade logs (candlestick charts)
3. **Sentiment** — LLM scoring results, headline analysis
4. **Ablation** — Component contribution radar charts
5. **Paper Trade** — Interactive strategy simulator

---

## Configuration

All parameters are centralized in `config/settings.yaml`:

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
```

---

## Testing

```bash
pytest tests/ -v
```

| Test Suite | Tests | Coverage |
|-----------|-------|---------|
| test_clean.py | 4 | Data cleaning & validation |
| test_features.py | 5 | Feature engineering & leakage prevention |
| test_backtest.py | 7 | Backtesting engine & metrics |
| test_env.py | 10 | RL environment & reward functions |

---

## Cloud Deployment

The dashboard auto-deploys to **Streamlit Community Cloud** ($0 cost):

1. Push repo to GitHub
2. Connect at [share.streamlit.io](https://share.streamlit.io)
3. Set `start/dashboard/app.py` as the main file
4. Add email whitelist in Streamlit Cloud Secrets

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

---

## Limitations

1. **Walk-forward accuracy ~50-52%** — Expected for next-bar prediction on financial data
2. **RL agents overfit in-sample** — Eval returns are lower than training returns
3. **Sentiment uses generic headlines** — Live news API (Marketaux) requires paid tier for full coverage
4. **Long-only** — No short selling or margin modeling
5. **Fixed position sizing** — 100 shares per trade, no dynamic allocation

---

## References

- Malo et al. (2014). "Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts." *Journal of the Association for Information Science and Technology.*
- Schulman et al. (2017). "Proximal Policy Optimization Algorithms." *arXiv:1707.06347.*
- Mnih et al. (2015). "Human-level control through deep reinforcement learning." *Nature.*

---

## Author

**Naresh Anbazhagan** — M.S. Data Analytics, McDaniel College

---

*Built with Python, PyTorch, stable-baselines3, scikit-learn, Streamlit, and Ollama.*
