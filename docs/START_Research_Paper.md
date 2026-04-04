# Strategic Technical Analysis for Reliable Trading (START): An Adaptive Multi-Interval Trading Framework Combining Machine Learning, Reinforcement Learning, and Real-Time Sentiment Analysis

**Naresh Anbazhagan**
M.S. Data Analytics, McDaniel College
Instructor: Dr. Xuejing Duan
April 2026

---

## Abstract

This paper presents START (Strategic Technical Analysis for Reliable Trading), an adaptive trading framework that integrates technical indicators, machine learning classifiers, deep learning architectures, reinforcement learning agents, and real-time sentiment analysis into a unified, cost-aware backtesting and live signal generation pipeline. The system evaluates 12 high-liquidity U.S. equities and ETFs across three time intervals (5-minute, hourly, daily) using walk-forward validation to prevent data leakage, with realistic transaction cost modeling including slippage (0.01%) and commissions ($0.005/share). A five-configuration ablation study isolates the contribution of each component. Results across 84 strategy-symbol combinations show that classical ML models (logistic regression, ridge regression) achieve the strongest risk-adjusted returns with mean Sharpe ratios of 2.00–2.07, outperforming random forest (1.00), buy-and-hold (1.78), and MA crossover (1.38) baselines. The AVGO PPO reinforcement learning agent achieved the highest individual Sharpe ratio of 6.02. Real-time sentiment analysis via Alpha Vantage NEWS_SENTIMENT API across 600 headlines reveals moderately bullish aggregate market sentiment (mean score: +0.149). The framework includes a live signal generation system combining five rule-based strategies with ML inference, deployed as an interactive Streamlit dashboard with real-time Tradier API price feeds. The 8,619-line codebase across 54 source files emphasizes reproducibility through centralized configuration (85 parameters), Parquet-based storage, and a single-command pipeline orchestrator.

**Keywords:** algorithmic trading, walk-forward validation, machine learning, reinforcement learning, sentiment analysis, cost-aware backtesting, ablation study, real-time trading signals, multi-interval analysis

---

## 1. Introduction

### 1.1 Motivation

Intraday trading in U.S. equity markets presents a challenging prediction problem. Price movements at short horizons are influenced by a complex interaction of technical patterns, fundamental factors, and market sentiment. Traditional rule-based trading strategies, while transparent, fail to adapt to changing market regimes. Conversely, machine learning approaches risk overfitting without rigorous temporal validation (Gu, Kelly, & Xiu, 2020).

The efficient market hypothesis (Fama, 1970) suggests that prices fully reflect available information, making consistent prediction theoretically impossible. However, extensive empirical evidence demonstrates that technical trading rules can generate statistically significant returns under specific conditions (Brock, Lakonishok, & LeBaron, 1992; Lo, Mamaysky, & Wang, 2000). The challenge lies in building systems that are rigorous enough to separate genuine signal from noise while accounting for the real costs of trading.

This paper addresses the question: **Can a hybrid framework combining technical indicators, ML classifiers, RL agents, and real-time sentiment analysis produce superior risk-adjusted returns compared to individual components, while maintaining statistical rigor through walk-forward validation and cost-aware backtesting?**

### 1.2 Research Questions

1. Do indicator+ML hybrids outperform simple rules on risk-adjusted returns across multiple time intervals?
2. Can reinforcement learning agents improve entry/exit timing while respecting drawdown limits?
3. Does real-time, symbol-specific sentiment analysis enhance signal quality compared to generic news?
4. Which component contributes most to overall performance, as measured by ablation analysis?
5. Can a live signal generation system provide actionable trading recommendations using current market data?

### 1.3 Contributions

- A modular, end-to-end trading research pipeline with 54 source files (8,619 LOC) across 7 modules
- Multi-interval analysis spanning 5-minute, hourly, and daily timeframes
- Walk-forward validation (expanding window) preventing temporal data leakage
- Cost-aware backtesting with realistic slippage and commission modeling
- Five-configuration ablation study isolating component contributions across all 12 symbols
- Real-time sentiment analysis using Alpha Vantage NEWS_SENTIMENT API (600 headlines analyzed)
- Live trading signal generation combining 5 rule-based strategies with ML inference
- Real-time market data integration via Tradier API with automatic daily aggregation
- SPY-based market regime classification (bull/bear/sideways)
- Interactive Streamlit dashboard with 7 pages deployed on Streamlit Community Cloud
- Single-command pipeline orchestrator with 85 centralized configuration parameters

---

## 2. Literature Review

### 2.1 Technical Analysis

Technical analysis has been debated extensively in financial literature. Brock, Lakonishok, and LeBaron (1992) demonstrated that simple technical trading rules — moving average crossovers and support/resistance breaks — generated statistically significant returns on the Dow Jones Industrial Average over 1897–1986, a finding that challenged the weak form of the efficient market hypothesis (Fama, 1970). Lo, Mamaysky, and Wang (2000) subsequently applied kernel regression to automate pattern recognition, finding that several technical patterns provided incremental predictive information when combined with conventional statistical methods. More recently, Park and Irwin (2007) conducted a comprehensive survey of 95 modern studies on technical analysis profitability, concluding that 56 studies found positive results, 20 negative, and 19 mixed, suggesting continued viability of technical approaches.

### 2.2 Machine Learning in Finance

Machine learning approaches to financial prediction have expanded significantly in the past decade. Gu, Kelly, and Xiu (2020) provided the definitive empirical comparison, testing neural networks, random forests, gradient boosting, and regularized linear models on the cross-section of U.S. equity returns. Their finding that tree-based models and neural networks outperform linear benchmarks catalyzed a wave of ML-based trading research. Fischer and Krauss (2018) demonstrated that LSTM networks capture temporal dependencies in daily price sequences for S&P 500 constituents, outperforming random forests and logistic regression in directional prediction. However, most studies fail to account for transaction costs or use look-ahead-biased evaluation methodologies — a problem highlighted by Bailey, Borwein, Lopez de Prado, and Zhu (2014), who demonstrated that multiple testing bias inflates apparent strategy performance.

### 2.3 Reinforcement Learning for Trading

Reinforcement learning for portfolio management and execution has gained attention through foundational work in deep RL. Mnih et al. (2015) introduced Deep Q-Networks (DQN), demonstrating human-level performance in Atari games through end-to-end learning from raw pixels. Schulman, Wolski, Dhariwal, Radford, and Klimov (2017) proposed Proximal Policy Optimization (PPO), a policy gradient method that achieves strong performance with simpler implementation than trust region methods. In the trading domain, Deng, Bao, Kong, Ren, and Dai (2017) applied deep reinforcement learning to financial signal representation and trading, showing that RL agents can learn profitable strategies by directly optimizing cumulative returns. These algorithms naturally incorporate transaction costs into the reward signal, aligning training objectives with practical trading constraints.

### 2.4 Sentiment Analysis in Finance

Sentiment analysis in finance has evolved from lexicon-based approaches (Loughran & McDonald, 2011) to transformer-based models. Malo, Sinha, Korhonen, Wallenius, and Takala (2014) created the Financial PhraseBank, a benchmark dataset of 4,845 financial news sentences annotated by 16 domain experts, which has become a standard evaluation corpus. More recently, large language models have demonstrated strong financial sentiment classification capabilities. Araci (2019) introduced FinBERT, a BERT model pre-trained on financial text, achieving state-of-the-art results on financial sentiment benchmarks. The Alpha Vantage NEWS_SENTIMENT API (Alpha Vantage, 2024) provides pre-computed sentiment scores for financial news using machine learning, enabling real-time sentiment integration without local model inference.

### 2.5 Backtesting Methodology

Rigorous backtesting is essential for evaluating trading strategies. Harvey, Liu, and Zhu (2016) demonstrated that the majority of claimed anomalies in published finance research are likely false discoveries due to multiple testing. De Prado (2018) formalized walk-forward validation as the correct approach for time series prediction, arguing that cross-validation violates temporal ordering assumptions. Our implementation follows these methodological guidelines, using expanding-window walk-forward validation with strictly out-of-sample evaluation periods.

---

## 3. Data

### 3.1 Universe

The evaluation universe consists of 12 high-liquidity U.S. equities and ETFs selected for market capitalization, trading volume, and sector diversity:

| Symbol | Description | Sector | Rationale |
|--------|-------------|--------|-----------|
| SPY | S&P 500 ETF | Index | Market benchmark |
| QQQ | Nasdaq-100 ETF | Index | Tech-heavy benchmark |
| NVDA | NVIDIA Corporation | Semiconductors | High volatility, AI leader |
| AAPL | Apple Inc. | Technology | Mega-cap, high liquidity |
| TSLA | Tesla Inc. | Consumer | Extreme volatility stress-test |
| MSFT | Microsoft Corporation | Technology | Stable mega-cap |
| AMZN | Amazon.com Inc. | Consumer | E-commerce leader |
| META | Meta Platforms Inc. | Technology | Social media leader |
| GOOGL | Alphabet Inc. | Technology | Search/advertising |
| AMD | Advanced Micro Devices | Semiconductors | High beta, momentum |
| NFLX | Netflix Inc. | Communication | Entertainment sector |
| AVGO | Broadcom Inc. | Semiconductors | Semiconductor diversification |

### 3.2 Data Sources and Multi-Interval Coverage

The framework implements a multi-source, multi-interval data architecture:

**Primary Source — yfinance (Yahoo Finance):** Hourly OHLCV bars covering approximately April 2024 through March 2026 (~3,344 bars per symbol). For intraday 5-minute data, yfinance provides the most recent 60 trading days (~1,737 bars per symbol) using the `period='60d'` parameter.

**Secondary Source — Tradier Markets API:** Real-time quotes for all 12 symbols via the `/v1/markets/quotes` endpoint, plus intraday 5-minute historical data via `/v1/markets/timesales` covering the most recent 5 weeks. Tradier provides authoritative real-time pricing used for the live signal dashboard.

**Daily Aggregation:** Daily bars are computed on-the-fly by aggregating hourly data — Open=first, High=max, Low=min, Close=last, Volume=sum — with technical indicators recalculated on the aggregated bars. This eliminates the need for a separate daily data source while ensuring consistency.

| Interval | Source | Bars/Symbol | Period |
|----------|--------|------------|--------|
| 5-minute | yfinance + Tradier | ~1,737 | 60 days |
| 1-hour | yfinance | ~3,344 | 2 years |
| 1-day | Aggregated from 1h | ~429 | 2 years |

**Total data points:** 40,128 hourly bars + 20,868 five-minute bars across the full universe.

### 3.3 Data Cleaning

The cleaning pipeline validates:
- **Session alignment**: Bars within regular trading hours (9:30 AM – 4:00 PM ET)
- **OHLCV consistency**: low ≤ open,close ≤ high; volume ≥ 0
- **Duplicate removal**: Earliest record retained per timestamp
- **Outlier detection**: Price spikes exceeding 8 standard deviations flagged
- **Timezone normalization**: All timestamps converted to US/Eastern, then stripped for cross-platform compatibility

### 3.4 Feature Engineering

Seventeen features are computed using TA-Lib (with pure-pandas fallback) across four categories:

| Feature | Description | Category |
|---------|-------------|----------|
| sma_20, sma_50 | Simple Moving Averages | Trend |
| rsi_14 | 14-period Relative Strength Index | Momentum |
| macd, macd_signal, macd_hist | MACD (12, 26, 9) | Momentum |
| bb_upper, bb_middle, bb_lower | Bollinger Bands (20, 2σ) | Volatility |
| bb_width, bb_pct | Band width and %B | Volatility |
| simple_return, log_return | Period returns | Returns |
| rolling_volatility | 5-bar rolling standard deviation | Volatility |
| cumulative_return | Cumulative return from start | Returns |
| intraday_range | (high − low) / close | Volatility |
| volume_ratio | Volume / 20-bar volume MA | Activity |

**Target variable**: Binary direction — 1 if next bar's close > current close, 0 otherwise. Constructed causally using `shift(-1)` to prevent look-ahead bias. The last bar of each symbol receives a NaN target and is excluded from training.

**Correlation management**: Features with absolute Pearson correlation > 0.95 can be optionally filtered at training time. The full 17-feature set is preserved in storage; correlation filtering is deferred to the model training step to allow different models to use different feature subsets.

---

## 4. Methodology

### 4.1 Walk-Forward Validation

All models are evaluated using expanding-window walk-forward validation following the methodology recommended by De Prado (2018). The procedure:

1. Train on bars [0, T] where T starts at 1,500 bars
2. Predict on bars [T, T+200]
3. Advance T by 200 bars
4. Repeat until data exhaustion

This yields approximately 9 out-of-sample folds per symbol (at the hourly interval), with all predictions strictly out-of-sample. No shuffling is performed at any stage to preserve temporal ordering.

### 4.2 Baseline Strategies

**Buy-and-Hold**: Signal = 1 at all times. Represents passive market exposure and serves as the primary benchmark.

**MA Crossover**: Signal = 1 when SMA(20) > SMA(50), 0 otherwise. A classic trend-following rule documented extensively in Brock et al. (1992).

### 4.3 Classical Machine Learning Models

All classical models use scikit-learn (Pedregosa et al., 2011) with StandardScaler preprocessing:

**Logistic Regression**: StandardScaler → LogisticRegression (C=1.0, max_iter=1000). Produces calibrated probability estimates suitable for signal confidence scoring.

**Ridge Regression**: StandardScaler → Ridge (α=1.0). Continuous output thresholded at 0.5 for binary classification. L2 regularization prevents overfitting on the limited feature set.

**Random Forest**: StandardScaler → RandomForestClassifier (n_estimators=50, max_depth=8, min_samples_leaf=20). Captures non-linear feature interactions with regularization through depth limiting and minimum leaf size.

### 4.4 Deep Learning Models

Both deep learning models use PyTorch (Paszke et al., 2019) with CPU-only inference (models <10K parameters; GPU provides no speedup):

**LSTM**: Single-layer LSTM (32 hidden units) with 20-bar input sequences (Hochreiter & Schmidhuber, 1997). BCEWithLogitsLoss, Adam optimizer (lr=0.001), early stopping (patience=5). Total parameters < 5,000.

**1D CNN**: Two convolutional layers (16 filters k=5, 8 filters k=3) with adaptive average pooling, following the temporal CNN architecture described by Bai, Kolter, and Koltun (2018). 20-bar input sequences with same training protocol as LSTM.

### 4.5 Reinforcement Learning

**Environment**: Custom Gymnasium (Towers et al., 2023) environment with discrete action space {Hold=0, Buy=1, Sell=2}. Long-only constraint (no short selling). Observation: flattened 20-bar feature window concatenated with position indicator. Features are z-normalized using training-set statistics.

**Reward**: After-cost return with drawdown penalty:

r_t = PnL_t − costs_t − λ · drawdown_t

where λ = 0.1 penalizes equity drawdowns, following the reward shaping approach described by Deng et al. (2017).

**PPO** (Schulman et al., 2017): MLP policy, lr=3×10⁻⁴, 256-step rollouts, 10 epochs per update, γ=0.99, GAE λ=0.95, clip range=0.2. Trained for 50,000 timesteps per symbol.

**DQN** (Mnih et al., 2015): MLP policy, lr=1×10⁻⁴, replay buffer=10,000, learning starts=500, ε-greedy exploration (30% fraction, final ε=0.05), target network update every 250 steps.

Both agents are implemented using Stable-Baselines3 (Raffin et al., 2021).

### 4.6 Sentiment Analysis

**Primary Source — Alpha Vantage NEWS_SENTIMENT API**: The Alpha Vantage NEWS_SENTIMENT endpoint (Alpha Vantage, 2024) provides pre-computed sentiment scores for financial news articles, eliminating the need for local LLM inference during production. For each of the 12 symbols, 50 recent headlines are fetched with symbol-specific relevance filtering. Each headline includes:
- Sentiment score: continuous value from −1 (bearish) to +1 (bullish)
- Relevance score: confidence that the article pertains to the queried symbol (0–1)
- Source attribution and publication timestamp

Weighted sentiment is computed as: weighted_score = sentiment × relevance.

**Fallback — Local LLM**: Mistral 7B Instruct (Q4_K_M quantization, ~4.1 GB) running locally via Ollama (Ollama, 2024) provides offline sentiment scoring when API access is unavailable.

**Academic Evaluation Corpus**: Financial PhraseBank (Malo et al., 2014) — 4,845 financial news sentences annotated by 16 domain experts — serves as the validation dataset for the LLM sentiment classifier.

### 4.7 Cost-Aware Backtesting

All strategies are backtested with realistic transaction costs following industry-standard modeling (Kissell, 2013):
- **Slippage**: 0.01% of trade price (applied on both entry and exit)
- **Commission**: $0.005 per share per trade
- **Position size**: 100 shares per trade
- **Initial capital**: $100,000

Performance metrics computed: net PnL, total return, annualized return, Sharpe ratio (Sharpe, 1994), Sortino ratio, Calmar ratio, maximum drawdown, win rate, profit factor, turnover, total costs.

### 4.8 Regime Analysis

Market regime is classified using SPY following the trend-based classification approach:
- **Bull**: SMA(50) slope > 0 AND price > SMA(50)
- **Bear**: SMA(50) slope < 0 AND price < SMA(50)
- **Sideways**: Neither condition met

Over the evaluation period, SPY regimes were: bull 45.5%, sideways 29.6%, bear 24.9%.

### 4.9 Ablation Study

Five configurations isolate each component's marginal contribution across all 12 symbols:

| Config | Components | Description |
|--------|-----------|-------------|
| buy_and_hold | Passive long | Market exposure baseline |
| indicators_only | MA crossover | Technical rules baseline |
| indicators_ml | TA + Logistic | Best classical ML model |
| indicators_ml_sentiment | TA + ML + Sentiment | Sentiment modulation |
| full_hybrid | Ensemble vote | Majority of all signal sources |

### 4.10 Live Signal Generation

A real-time signal generation system produces actionable trading recommendations by combining five rule-based strategies with optional ML inference:

1. **MA Crossover**: SMA(20) vs SMA(50) with crossover event detection
2. **RSI Mean Reversion**: Oversold (<30) / Overbought (>70) thresholds
3. **MACD Momentum**: MACD line vs signal line with histogram strength
4. **Bollinger Band Position**: %B relative to upper/lower bands
5. **ML Ensemble**: Logistic, Ridge, and Random Forest trained on all-but-latest bar

Consensus is computed using confidence-weighted voting across all strategies. Each strategy produces a signal (BUY=1, SELL=0, HOLD=−1) with an associated confidence score (0–1). The overall recommendation is determined by the side with the highest cumulative confidence weight.

---

## 5. Results

### 5.1 Classical ML Performance

Walk-forward validation across 12 symbols (expanding window, 1,500-bar train / 200-bar test, hourly interval):

| Strategy | Mean Sharpe | Std | Best Symbol | Worst Symbol |
|----------|:-----------:|:---:|:-----------:|:------------:|
| Ridge Regression | **2.07** | 2.36 | AMD (5.38) | MSFT (−2.86) |
| Logistic Regression | **2.00** | 2.31 | AMD (5.21) | MSFT (−2.42) |
| Buy-and-Hold | 1.78 | 1.10 | TSLA (3.62) | MSFT (−1.23) |
| MA Crossover | 1.38 | 1.13 | SPY (3.13) | MSFT (−2.58) |
| Random Forest | 1.00 | 1.15 | NVDA (1.84) | MSFT (−3.69) |

**Key finding**: Ridge and logistic regression achieve the highest mean Sharpe ratios, outperforming both baselines and the more complex random forest. This is consistent with the bias-variance tradeoff analysis by Gu et al. (2020) — at hourly frequency with ~3,300 bars, complex models lack sufficient data to learn stable patterns.

### 5.2 Top Performing Symbol-Strategy Pairs

| Symbol | Strategy | Sharpe | Return | Max DD | Win Rate | Trades |
|--------|----------|:------:|:------:|:------:|:--------:|:------:|
| AMD | Ridge | 5.38 | 18.4% | 2.8% | 58.4% | 8 |
| AMD | Logistic | 5.21 | 17.8% | 2.6% | 57.4% | 8 |
| META | Logistic | 4.46 | 24.6% | 5.5% | 56.2% | 8 |
| META | Ridge | 4.30 | 24.8% | 5.6% | 56.2% | 8 |
| AMZN | Logistic | 3.78 | 8.7% | 3.6% | 61.9% | 9 |
| AMZN | Ridge | 3.73 | 8.5% | 3.5% | 62.3% | 9 |
| GOOGL | Ridge | 3.49 | 9.3% | 3.1% | 60.1% | 7 |
| NVDA | Ridge | 3.58 | 7.8% | 3.5% | 59.1% | 8 |
| TSLA | Ridge | 3.46 | 15.6% | 6.5% | 58.8% | 7 |

### 5.3 Performance by Symbol

Mean Sharpe ratio across all 5 strategies per symbol:

| Symbol | Mean Sharpe | Interpretation |
|--------|:-----------:|---------------|
| AMD | 2.78 | Strongest alpha generation across strategies |
| GOOGL | 2.67 | Consistent predictability across all approaches |
| NVDA | 2.38 | Good signal-to-noise ratio |
| META | 2.24 | High returns with moderate risk |
| AMZN | 2.22 | Highest win rates (61–62%) |
| SPY | 2.15 | Index-level, reliable benchmark returns |
| TSLA | 2.05 | High volatility enables trend-following |
| QQQ | 1.62 | Moderate signal strength |
| AVGO | 1.44 | Moderate, benefits from RL |
| AAPL | 1.17 | Lower alpha in ML, adequate in buy-and-hold |
| NFLX | 0.54 | Weak predictive signal |
| MSFT | −1.49 | Negative alpha — challenging for all strategies |

### 5.4 Reinforcement Learning

PPO and DQN agents trained with 50,000 timesteps per symbol using Stable-Baselines3 (Raffin et al., 2021):

| Symbol | Agent | Sharpe | Return | Max DD | Trades |
|--------|-------|:------:|:------:|:------:|:------:|
| AVGO | PPO | **6.02** | 29.8% | 4.5% | 12 |
| QQQ | PPO | 3.60 | 12.9% | 3.8% | 9 |
| TSLA | PPO | 3.56 | 21.1% | 7.2% | 14 |
| META | PPO | 3.35 | 25.9% | 8.1% | 11 |
| AMD | DQN | 2.26 | 8.8% | 4.1% | 18 |

**Key observations**:
- PPO consistently outperforms DQN, likely due to better sample efficiency with limited training timesteps
- PPO agents achieve the single highest Sharpe ratio in the study (AVGO: 6.02)
- DQN agents exhibit higher trade frequency but more variable returns
- High variance across symbols suggests RL benefits from longer training horizons and curriculum learning

### 5.5 Sentiment Analysis

Alpha Vantage NEWS_SENTIMENT API analysis across 600 headlines (50 per symbol):

| Symbol | Mean Sentiment | Confidence | Positive % | Negative % |
|--------|:--------------:|:----------:|:----------:|:----------:|
| AMD | +0.247 | 0.773 | 94% | 6% |
| AVGO | +0.227 | 0.809 | 92% | 8% |
| MSFT | +0.208 | 0.650 | 94% | 6% |
| AMZN | +0.172 | 0.733 | 88% | 12% |
| NVDA | +0.164 | 0.657 | 82% | 18% |
| META | +0.154 | 0.745 | 78% | 22% |
| NFLX | +0.146 | 0.725 | 82% | 18% |
| AAPL | +0.136 | 0.756 | 74% | 26% |
| GOOGL | +0.125 | 0.730 | 80% | 20% |
| SPY | +0.095 | 0.897 | 74% | 26% |
| QQQ | +0.077 | 0.696 | 86% | 14% |
| TSLA | +0.032 | 0.737 | 66% | 34% |

**Aggregate statistics**: Mean sentiment = +0.149 (moderately bullish), mean confidence = 0.742, total headlines = 600.

**Key finding**: Symbol-specific sentiment from Alpha Vantage provides meaningful differentiation — AMD (+0.247) is the most bullish while TSLA (+0.032) is nearly neutral, reflecting the polarized market opinions on Tesla. The high confidence scores (0.65–0.90) indicate the API's ML-based scoring is decisive rather than uncertain.

### 5.6 Ablation Study

Ablation results across the full 12-symbol universe (60 configurations total):

| Configuration | Mean Sharpe | Mean Return | Mean Max DD | Mean Win Rate |
|--------------|:-----------:|:-----------:|:-----------:|:-------------:|
| indicators_ml | **2.23** | 5.8% | 4.2% | 58.5% |
| indicators_ml_sentiment | 2.23 | 5.8% | 4.2% | 58.5% |
| buy_and_hold | 1.78 | 7.8% | 7.9% | — |
| full_hybrid | 1.65 | 4.1% | 3.8% | 55.2% |
| indicators_only | 1.38 | 2.5% | 4.6% | 48.5% |

**Key findings**:
1. **ML adds measurable alpha**: indicators_ml improves Sharpe by +0.85 over indicators_only
2. **Sentiment integration requires refinement**: indicators_ml_sentiment shows identical performance to indicators_ml, suggesting that the current weighted-average fusion method does not effectively incorporate sentiment signals. Alternative approaches — such as sentiment-conditional strategy selection or regime-dependent weighting — may unlock the value of sentiment data
3. **Hybrid ensemble underperforms**: The majority-vote ensemble achieves lower Sharpe than indicators_ml alone, suggesting that averaging heterogeneous signals introduces noise rather than diversification
4. **Buy-and-hold achieves higher raw returns**: The 7.8% return exceeds indicators_ml's 5.8%, but at significantly higher drawdown (7.9% vs 4.2%), resulting in inferior risk-adjusted performance

---

## 6. Discussion

### 6.1 Why Simple Models Win

Ridge and logistic regression outperform random forest, LSTM, and CNN across the symbol universe. This aligns with the findings of Gu et al. (2020) regarding the bias-variance tradeoff: at hourly frequency with ~3,300 bars per symbol, complex models with many free parameters lack sufficient data to learn stable patterns. Regularized linear models efficiently extract the available linear signal without overfitting to noise. The random forest's inferior performance (mean Sharpe 1.00 vs 2.07 for ridge) suggests that non-linear feature interactions at this time scale are either absent or too unstable to capture reliably with limited data.

### 6.2 The 51% Accuracy Paradox

Classification accuracy of ~51% may seem marginal, but two factors explain the positive risk-adjusted returns. First, as discussed by Lopez de Prado (2018), accuracy is a poor metric for trading — what matters is the asymmetric payoff structure. A model that correctly identifies large moves while making small errors on flat periods generates positive returns despite near-random accuracy. Second, the 58–62% win rates in backtesting (higher than prediction accuracy) reflect the cost-aware execution filter: the backtesting engine's signal-to-trade conversion eliminates many marginal positions, retaining only trades where the signal persists long enough to overcome transaction costs.

### 6.3 Multi-Interval Analysis

The three-interval architecture (5-minute, hourly, daily) serves distinct analytical purposes:
- **5-minute**: Captures intraday momentum and mean-reversion patterns within 60-day windows
- **Hourly**: Primary analysis interval with 2 years of data — sufficient for walk-forward validation
- **Daily**: Derived from hourly data via aggregation; reveals longer-term trends and regime shifts

The daily aggregation approach (computing OHLCV from hourly bars, then recalculating indicators) ensures consistency across intervals without requiring separate data ingestion.

### 6.4 Live Signal System

The live signal generation system bridges the gap between historical backtesting and real-time decision support. By combining five rule-based strategies (MA Crossover, RSI, MACD, Bollinger Bands) with optional ML inference, the system provides a confidence-weighted consensus recommendation. Initial results across all 12 symbols show:
- Consensus agreement ranges from 40–60% across symbols
- Higher agreement correlates with stronger trending conditions
- The system correctly identifies RSI oversold conditions (e.g., AAPL RSI=29.5 → BUY signal) and MACD bearish divergences

### 6.5 Regime Effects

SPY regime analysis reveals that model performance varies significantly across market conditions. Bull regimes (45.5% of the period) naturally favor long-biased strategies including buy-and-hold. The framework's regime classification enables future conditional strategy selection, though this paper evaluates strategies unconditionally.

### 6.6 Limitations

1. **Evaluation period**: ~2 years of hourly data limits statistical power for regime-conditional analysis and multi-cycle evaluation
2. **Sentiment fusion**: The current weighted-average approach does not effectively incorporate sentiment — alternative architectures (attention-based fusion, sentiment-conditional gating) may be needed
3. **RL training budget**: 50,000 timesteps is relatively short for complex trading environments; longer training with curriculum learning (Bengio, Louradour, Collobert, & Weston, 2009) may improve stability
4. **Long-only constraint**: Restricting to long/flat positions limits alpha generation in bear markets
5. **Fixed position sizing**: 100 shares per trade does not adapt to volatility or conviction
6. **Single-asset evaluation**: Each model trades one symbol independently; portfolio-level optimization (Markowitz, 1952) is not addressed
7. **MSFT anomaly**: MSFT shows consistently negative alpha across all strategies, suggesting period-specific market microstructure factors that warrant separate investigation
8. **Live signal validation**: The live signal system has not been validated through paper trading over a statistically significant period

---

## 7. Conclusion

START demonstrates that a well-engineered pipeline with rigorous temporal validation can produce meaningful trading signals from multi-interval equity data. The key findings are:

1. **Simple ML models outperform complex ones** at hourly frequency with limited data — regularized linear models (ridge, logistic) achieve Sharpe ratios of 2.0+ averaged across 12 symbols, consistent with the bias-variance analysis of Gu et al. (2020)
2. **Reinforcement learning achieves the highest individual performance** — PPO on AVGO achieves a Sharpe ratio of 6.02, demonstrating the potential of RL-based trading when matched with the right market dynamics
3. **Walk-forward validation is essential** — the ~51% accuracy reflects genuine out-of-sample performance, avoiding the inflated metrics common in financial ML research (Bailey et al., 2014)
4. **Cost-aware backtesting reveals true performance** — strategies that appear profitable before costs may underperform after realistic transaction cost modeling
5. **Symbol-specific sentiment provides differentiation** — Alpha Vantage NEWS_SENTIMENT scores range from +0.032 (TSLA) to +0.247 (AMD), providing meaningful cross-sectional signal, though effective fusion into trading strategies remains an open challenge
6. **Ablation studies quantify component contributions** — ML adds +0.85 Sharpe over technical baselines, while sentiment and ensemble contributions depend on fusion methodology
7. **Multi-interval analysis enables diverse applications** — the 5-minute to daily spectrum serves different trading horizons from the same underlying data pipeline
8. **Reproducibility through engineering** — 85 centralized configuration parameters, Parquet storage, and single-command execution ensure results can be replicated

### 7.1 Future Work

- **Sentiment fusion architectures**: Attention-based or gating mechanisms for sentiment-ML integration
- **Portfolio optimization**: Multi-asset allocation using Modern Portfolio Theory (Markowitz, 1952) or risk parity
- **Extended RL training**: Curriculum learning with 500K+ timesteps and hyperparameter search
- **Cross-interval signals**: Using daily regime to condition intraday 5-minute strategies
- **Paper trading validation**: Forward-testing live signals over 3–6 months
- **Alternative data**: Incorporating options flow, insider transactions, or social media sentiment

---

## 8. References

Alpha Vantage. (2024). NEWS_SENTIMENT API documentation. https://www.alphavantage.co/documentation/

Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. *arXiv preprint arXiv:1908.10063*.

Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of generic convolutional and recurrent networks for sequence modeling. *arXiv preprint arXiv:1803.01271*.

Bailey, D. H., Borwein, J. M., Lopez de Prado, M., & Zhu, Q. J. (2014). Pseudomathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance. *Notices of the American Mathematical Society*, 61(5), 458–471.

Bengio, Y., Louradour, J., Collobert, R., & Weston, J. (2009). Curriculum learning. *Proceedings of the 26th International Conference on Machine Learning*, 41–48.

Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *The Journal of Finance*, 47(5), 1731–1764.

De Prado, M. L. (2018). *Advances in Financial Machine Learning*. John Wiley & Sons.

Deng, Y., Bao, F., Kong, Y., Ren, Z., & Dai, Q. (2017). Deep direct reinforcement learning for financial signal representation and trading. *IEEE Transactions on Neural Networks and Learning Systems*, 28(3), 653–664.

Fama, E. F. (1970). Efficient capital markets: A review of theory and empirical work. *The Journal of Finance*, 25(2), 383–417.

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223–2273.

Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *The Review of Financial Studies*, 29(1), 5–68.

Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.

Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management*. Academic Press.

Lo, A. W., Mamaysky, H., & Wang, J. (2000). Foundations of technical analysis: Computational algorithms, statistical inference, and empirical implementation. *The Journal of Finance*, 55(4), 1705–1765.

Loughran, T., & McDonald, B. (2011). When is a liability not a liability? Textual analysis, dictionaries, and 10-Ks. *The Journal of Finance*, 66(1), 35–65.

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.

Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77–91.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.

Ollama. (2024). Ollama: Run large language models locally. https://ollama.com

Park, C.-H., & Irwin, S. H. (2007). What do we know about the profitability of technical analysis? *Journal of Economic Surveys*, 21(4), 786–826.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., ... & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.

Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., & Dormann, N. (2021). Stable-Baselines3: Reliable reinforcement learning implementations. *Journal of Machine Learning Research*, 22(268), 1–8.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

Sharpe, W. F. (1994). The Sharpe ratio. *The Journal of Portfolio Management*, 21(1), 49–58.

Towers, M., Terry, J. K., Kwiatkowski, A., Balis, J. U., Cola, G. D., Deleu, T., ... & Younis, O. G. (2023). Gymnasium. https://gymnasium.farama.org

---

## Appendix A: System Architecture

```
run_pipeline.py (orchestrator)
│
├── 01_data_ingestion.py      → data/raw/ → data/parquet/
│   └── Providers: yfinance (primary), Tradier (real-time), Alpaca (optional)
├── 02_feature_engineering.py  → data/features/ (5min + 1h Parquet per symbol)
├── 03_model_training.py       → data/results/model_comparison.parquet
├── 04_reinforcement_learning.py → data/models/*.zip, data/results/rl_comparison.parquet
├── 05_sentiment_analysis.py   → data/sentiment/, data/results/sentiment_scores.parquet
│   └── Sources: Alpha Vantage API → Marketaux → Financial PhraseBank
├── 06_ablation_backtest.py    → data/results/ablation_results.parquet
└── 07_dashboard_launcher.py   → Streamlit app (7 pages, reads all results)
    └── Live: Tradier quotes + live signal generation
```

## Appendix B: Configuration Parameters

All 85 tunable parameters are centralized in `config/settings.yaml`. Key parameters:

| Category | Parameter | Value |
|----------|-----------|-------|
| Walk-Forward | train_bars | 1,500 |
| Walk-Forward | test_bars | 200 |
| Walk-Forward | mode | expanding |
| Logistic | C | 1.0 |
| Ridge | α | 1.0 |
| Random Forest | n_estimators | 50 |
| Random Forest | max_depth | 8 |
| LSTM | hidden_size | 32 |
| LSTM/CNN | seq_len | 20 |
| LSTM/CNN | max_epochs | 30 |
| PPO | learning_rate | 3×10⁻⁴ |
| PPO | n_steps | 256 |
| DQN | learning_rate | 1×10⁻⁴ |
| DQN | buffer_size | 10,000 |
| RL | total_timesteps | 50,000 |
| Backtest | slippage | 0.01% |
| Backtest | commission | $0.005/share |
| Backtest | initial_capital | $100,000 |
| Sentiment | headlines_per_symbol | 50 |
| Features | corr_threshold | 0.95 |

## Appendix C: Dashboard Pages

| Page | Description | Key Visualizations |
|------|-------------|-------------------|
| Home | Project overview + live ticker | Real-time Tradier price feed (12 symbols) |
| Signal Heatmap | Cross-strategy comparison | Bubble chart, heatmap, treemap, parallel coordinates |
| PnL Charts | Equity curves and price analysis | Candlestick + BB/RSI/MACD panels, drawdown, return distribution |
| Sentiment | News sentiment analysis | Gauge meter, lollipop chart, headline explorer |
| Ablation | Component contribution analysis | Animated bar chart, radar chart, box plots |
| Paper Trade | Interactive strategy simulator | Equity curve with buy/sell markers, trade waterfall |
| Decisions | Plain-English recommendations | Live signals, risk quiz, investment calculator |

## Appendix D: Test Suite

26 unit tests across 4 test files verify core functionality:

| Test File | Tests | Coverage |
|-----------|:-----:|----------|
| test_clean.py | 4 | Data cleaning, deduplication, empty handling |
| test_features.py | 5 | Indicators, returns, leakage prevention, correlation filter |
| test_backtest.py | 7 | Engine, metrics, drawdown, cost modeling |
| test_env.py | 10 | RL environment, reward functions, episode termination |
