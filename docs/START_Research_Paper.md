# Strategic Technical Analysis for Reliable Trading (START): An Adaptive Intraday Trading Framework Combining Machine Learning, Reinforcement Learning, and LLM-Based Sentiment Analysis

**Naresh Anbazhagan**
M.S. Data Analytics, McDaniel College
Instructor: Dr. Xuejing Duan
April 2026

---

## Abstract

This paper presents START (Strategic Technical Analysis for Reliable Trading), an adaptive intraday trading framework that integrates technical indicators, machine learning classifiers, deep learning architectures, reinforcement learning agents, and large language model (LLM)-based sentiment analysis into a unified, cost-aware backtesting pipeline. The system is evaluated on a universe of 12 high-liquidity U.S. equities and ETFs using walk-forward validation to prevent data leakage, with realistic transaction cost modeling including slippage (0.01%) and commissions ($0.005/share). A five-configuration ablation study isolates the contribution of each component. Results show that classical ML models (logistic regression, ridge regression) achieve the strongest risk-adjusted returns with mean Sharpe ratios of 2.00–2.07, outperforming random forest (1.00), buy-and-hold (1.78), and MA crossover (1.38) baselines across the evaluation period. PPO and DQN reinforcement learning agents demonstrate adaptive behavior but exhibit higher variance. LLM sentiment analysis using Mistral 7B provides directionally accurate headline classification. The framework emphasizes reproducibility through centralized configuration, Parquet-based storage, and a single-command pipeline orchestrator.

**Keywords:** algorithmic trading, walk-forward validation, machine learning, reinforcement learning, sentiment analysis, cost-aware backtesting, ablation study

---

## 1. Introduction

### 1.1 Motivation

Intraday trading in U.S. equity markets presents a challenging prediction problem. Price movements at short horizons are influenced by a complex interaction of technical patterns, fundamental factors, and market sentiment. Traditional rule-based trading strategies, while transparent, fail to adapt to changing market regimes. Conversely, machine learning approaches risk overfitting without rigorous temporal validation.

This paper addresses the question: **Can a hybrid framework combining technical indicators, ML classifiers, RL agents, and LLM sentiment analysis produce superior risk-adjusted returns compared to individual components, while maintaining statistical rigor through walk-forward validation and cost-aware backtesting?**

### 1.2 Research Questions

1. Do indicator+ML hybrids outperform simple rules on risk-adjusted intraday returns?
2. Can reinforcement learning agents improve entry/exit timing while respecting drawdown limits?
3. Does LLM-based sentiment alignment enhance robustness during news-driven regimes?
4. Which component contributes most to overall performance, as measured by ablation analysis?

### 1.3 Contributions

- A modular, end-to-end trading research pipeline with 54 source files across 7 modules
- Walk-forward validation (expanding window) preventing temporal data leakage
- Cost-aware backtesting with realistic slippage and commission modeling
- Five-configuration ablation study isolating component contributions
- Local LLM sentiment analysis using Mistral 7B via Ollama
- SPY-based market regime classification (bull/bear/sideways)
- Interactive Streamlit dashboard for result exploration
- Single-command pipeline orchestrator for full reproducibility

---

## 2. Literature Review

Technical analysis has been debated extensively in financial literature. Brock, Lakonishok, and LeBaron (1992) demonstrated that simple technical trading rules (moving average crossovers, support/resistance) generated statistically significant returns on the Dow Jones Industrial Average over 1897–1986. Lo, Mamaysky, and Wang (2000) applied kernel regression to automate pattern recognition, finding that several technical patterns provided incremental information when combined with conventional statistics.

Machine learning approaches to financial prediction have expanded significantly. Random forests and gradient boosting have shown promise in cross-sectional return prediction (Gu, Kelly, & Xiu, 2020). LSTM networks capture temporal dependencies in price sequences (Fischer & Krauss, 2018). However, most studies fail to account for transaction costs or use look-ahead-biased evaluation.

Reinforcement learning for trading has gained attention through the work of Mnih et al. (2015) on Deep Q-Networks and Schulman et al. (2017) on Proximal Policy Optimization. These algorithms learn optimal policies through interaction with an environment, naturally incorporating transaction costs into the reward signal.

Sentiment analysis in finance has evolved from dictionary-based approaches to transformer models. Recent work demonstrates that large language models can classify financial sentiment with accuracy comparable to specialized models (Malo et al., 2014), with the advantage of zero-shot capability.

---

## 3. Data

### 3.1 Universe

The evaluation universe consists of 12 high-liquidity U.S. equities and ETFs (Ring 1):

| Symbol | Description | Sector |
|--------|-------------|--------|
| SPY | S&P 500 ETF | Index |
| QQQ | Nasdaq-100 ETF | Index |
| NVDA | NVIDIA Corporation | Technology |
| AAPL | Apple Inc. | Technology |
| TSLA | Tesla Inc. | Consumer |
| MSFT | Microsoft Corporation | Technology |
| AMZN | Amazon.com Inc. | Consumer |
| META | Meta Platforms Inc. | Technology |
| GOOGL | Alphabet Inc. | Technology |
| AMD | Advanced Micro Devices | Technology |
| NFLX | Netflix Inc. | Communication |
| AVGO | Broadcom Inc. | Technology |

### 3.2 Data Source and Period

Hourly OHLCV bars were sourced from yfinance (Yahoo Finance API) covering approximately April 2024 through March 2026. The multi-source data ingestion layer supports Alpaca and Tradier APIs for future expansion.

### 3.3 Data Cleaning

The cleaning pipeline validates:
- **Session alignment**: Bars within regular trading hours (9:30 AM – 4:00 PM ET)
- **OHLCV consistency**: low ≤ open,close ≤ high; volume ≥ 0
- **Duplicate removal**: Earliest record retained per timestamp
- **Outlier detection**: Price spikes exceeding 8 standard deviations flagged

After cleaning, each symbol yields approximately 3,300 hourly bars.

### 3.4 Feature Engineering

Ten to eleven features survive correlation filtering (threshold |r| > 0.95):

| Feature | Description | Category |
|---------|-------------|----------|
| sma_20 | 20-period Simple Moving Average | Trend |
| rsi_14 | 14-period Relative Strength Index | Momentum |
| macd | MACD Line (12, 26) | Momentum |
| macd_hist | MACD Histogram | Momentum |
| bb_width | Bollinger Band Width (20, 2) | Volatility |
| bb_pct | Bollinger Band %B | Volatility |
| simple_return | One-bar simple return | Returns |
| rolling_volatility | 5-bar rolling standard deviation | Volatility |
| intraday_range | (high - low) / close | Volatility |
| volume_ratio | Volume / 20-bar volume MA | Activity |

**Target variable**: Binary direction — 1 if next bar's close > current close, 0 otherwise. The target is constructed causally using `shift(-1)` to prevent look-ahead bias.

**Correlation filtering**: Features with absolute Pearson correlation > 0.95 are automatically removed. Typically 6–7 redundant features are dropped (sma_50, bb_upper, bb_middle, bb_lower, log_return, cumulative_return).

---

## 4. Methodology

### 4.1 Walk-Forward Validation

All models are evaluated using expanding-window walk-forward validation to respect temporal ordering. The procedure:

1. Train on bars [0, T] where T starts at 1,500 bars
2. Predict on bars [T, T+200]
3. Advance T by 200 bars
4. Repeat until data exhaustion

This yields approximately 9 out-of-sample folds per symbol, with all predictions strictly out-of-sample. No shuffling is performed at any stage.

### 4.2 Baseline Strategies

**Buy-and-Hold**: Signal = 1 at all times. Represents passive market exposure.

**MA Crossover**: Signal = 1 when SMA(20) > SMA(50), 0 otherwise. A classic trend-following rule.

### 4.3 Classical Machine Learning Models

**Logistic Regression**: StandardScaler → LogisticRegression (C=1.0). Produces calibrated probability estimates.

**Ridge Regression**: StandardScaler → Ridge (α=1.0). Continuous output thresholded at 0.5 for binary classification.

**Random Forest**: StandardScaler → RandomForestClassifier (n_estimators=50, max_depth=8, min_samples_leaf=20). Captures non-linear feature interactions.

### 4.4 Deep Learning Models

**LSTM**: Single-layer LSTM (32 hidden units) with 20-bar input sequences. BCEWithLogitsLoss, Adam optimizer (lr=0.001), early stopping (patience=5). Total parameters < 5,000.

**1D CNN**: Two convolutional layers (16 filters k=5, 8 filters k=3) with adaptive average pooling. 20-bar input sequences. Same training protocol as LSTM.

### 4.5 Reinforcement Learning

**Environment**: Custom Gymnasium environment with discrete action space {Hold=0, Buy=1, Sell=2}. Long-only (no short selling). Observation: flattened 20-bar feature window + position indicator.

**Reward**: After-cost return with drawdown penalty:
$$r_t = \text{PnL}_t - \text{costs}_t - \lambda \cdot \text{drawdown}_t$$
where λ = 0.1 penalizes equity drawdowns.

**PPO** (Proximal Policy Optimization): MLP policy, lr=3×10⁻⁴, 256-step rollouts, 10 epochs per update, γ=0.99, clip range=0.2.

**DQN** (Deep Q-Network): MLP policy, lr=1×10⁻⁴, replay buffer=10,000, ε-greedy exploration (30% fraction, final ε=0.05).

### 4.6 Sentiment Analysis

**Model**: Mistral 7B Instruct (Q4_K_M quantization, ~4.1 GB) running locally via Ollama.

**Headline Source**: Financial PhraseBank (Malo et al., 2014) — an academic dataset of 4,845 financial news sentences annotated by 16 domain experts. Used as the evaluation corpus.

**Scoring**: Each headline is classified as positive (+1), neutral (0), or negative (−1) with a confidence score (0–1). Aggregate weighted sentiment is computed per symbol.

### 4.7 Cost-Aware Backtesting

All strategies are backtested with:
- **Slippage**: 0.01% of trade price (applied on both entry and exit)
- **Commission**: $0.005 per share per trade
- **Position size**: 100 shares per trade
- **Initial capital**: $100,000

Performance metrics computed: net PnL, total return, Sharpe ratio, Sortino ratio, Calmar ratio, maximum drawdown, win rate, profit factor, turnover.

### 4.8 Regime Analysis

Market regime is classified using SPY:
- **Bull**: SMA(50) slope > 0 AND price > SMA(50)
- **Bear**: SMA(50) slope < 0 AND price < SMA(50)
- **Sideways**: Neither condition met

Over the evaluation period, SPY regimes were: bull 45.5%, sideways 29.6%, bear 24.9%.

### 4.9 Ablation Study

Five configurations isolate each component's marginal contribution:

| Config | Components | Description |
|--------|-----------|-------------|
| indicators_only | MA crossover | Technical rules baseline |
| indicators_ml | TA + Logistic | Best classical ML model |
| indicators_ml_sentiment | TA + ML + LLM | Sentiment modulation |
| rl_only | PPO agent | Pure RL signals |
| full_hybrid | Ensemble vote | Majority of all signal sources |

---

## 5. Results

### 5.1 Classical ML Performance

Walk-forward validation across 12 symbols (expanding window, 1,500 train / 200 test bars):

| Strategy | Mean Sharpe | Std | Mean Return | Mean Accuracy |
|----------|:-----------:|:---:|:-----------:|:-------------:|
| Ridge Regression | **2.07** | 2.36 | — | 51.0% |
| Logistic Regression | **2.00** | 2.31 | — | 51.0% |
| Buy-and-Hold | 1.78 | 1.10 | — | — |
| MA Crossover | 1.38 | 1.13 | — | — |
| Random Forest | 1.00 | 1.15 | — | 49.8% |

**Key finding**: Ridge and logistic regression achieve the highest mean Sharpe ratios, outperforming both baselines and the more complex random forest. The prediction accuracy of ~51% translates to positive risk-adjusted returns through favorable skew in correct predictions.

### 5.2 Top Performing Symbol-Strategy Pairs

| Symbol | Strategy | Sharpe | Return | Max DD | Win Rate |
|--------|----------|:------:|:------:|:------:|:--------:|
| AMD | Ridge | 5.38 | 18.4% | 2.8% | 58.4% |
| AMD | Logistic | 5.21 | 17.8% | 2.6% | 57.4% |
| META | Logistic | 4.46 | 24.6% | 5.5% | 56.2% |
| META | Ridge | 4.30 | 24.8% | 5.6% | 56.2% |
| AMZN | Logistic | 3.78 | 8.7% | 3.6% | 61.9% |
| AMZN | Ridge | 3.73 | 8.5% | 3.5% | 62.3% |
| GOOGL | Ridge | 3.49 | 9.3% | 3.1% | 60.1% |
| TSLA | Ridge | 3.46 | 15.6% | 6.5% | 58.8% |

### 5.3 Performance by Symbol

Mean Sharpe ratio across all strategies per symbol:

| Symbol | Mean Sharpe | Interpretation |
|--------|:-----------:|---------------|
| AMD | 2.78 | Strongest alpha generation |
| GOOGL | 2.67 | Consistent across strategies |
| NVDA | 2.38 | Good predictability |
| META | 2.24 | High return, moderate risk |
| AMZN | 2.22 | High win rate signals |
| SPY | 2.15 | Index-level returns |
| TSLA | 2.05 | High volatility, high reward |
| QQQ | 1.62 | Moderate |
| AVGO | 1.44 | Moderate |
| AAPL | 1.17 | Lower alpha |
| NFLX | 0.54 | Weak signal |
| MSFT | −1.49 | Negative alpha (challenging) |

### 5.4 Reinforcement Learning

PPO and DQN agents trained with 20,000 timesteps per symbol. Results for the full 12-symbol universe show:
- PPO agents demonstrated adaptive behavior with variable action distributions (hold, buy, sell)
- DQN agents showed higher trade frequency but more variable returns
- Both agents showed positive eval-period returns for several symbols (META PPO: +39%, AVGO DQN: +28%, TSLA DQN: +24%)
- High variance across symbols suggests RL benefits from longer training horizons

### 5.5 Sentiment Analysis

Mistral 7B achieved directionally accurate classification on Financial PhraseBank headlines:
- Positive headlines (e.g., "record quarterly earnings"): correctly classified as +1
- Negative headlines (e.g., "profit warning, supply disruptions"): correctly classified as −1
- Neutral headlines (e.g., "results in line with expectations"): correctly classified as 0

Mean confidence scores ranged from 0.80–0.95, indicating high model certainty. The balanced dataset (33% positive, 33% negative, 33% neutral) resulted in near-zero aggregate sentiment scores, as expected.

### 5.6 Ablation Study

Ablation results across the evaluation period (averaged across symbols with available configurations):

| Configuration | Sharpe | Return | Max DD | Win Rate |
|--------------|:------:|:------:|:------:|:--------:|
| buy_and_hold | 1.78 | 7.8% | 7.9% | — |
| indicators_only | 1.38 | 2.5% | 4.6% | 48.5% |
| indicators_ml | 2.00 | — | — | 58–62% |
| indicators_ml_sentiment | 2.00 | — | — | 58–62% |
| rl_only | Variable | Variable | Variable | 50–58% |
| full_hybrid | Variable | — | — | 55–58% |

**Key findings**:
1. ML models (logistic/ridge) consistently improve upon technical-only baselines
2. Sentiment has minimal impact when using PhraseBank (balanced dataset); live news would provide directional signal
3. RL agents add diversity but increase variance
4. The hybrid ensemble does not uniformly outperform individual components, suggesting that simple averaging of heterogeneous signals introduces noise

---

## 6. Discussion

### 6.1 Why Simple Models Win

Ridge and logistic regression outperform random forest, LSTM, and CNN. This is consistent with the bias-variance tradeoff: at hourly frequency with ~3,300 bars, complex models lack sufficient data to learn stable patterns. Regularized linear models extract the available signal without overfitting to noise.

### 6.2 The 51% Accuracy Paradox

Classification accuracy of ~51% may seem marginal, but the key insight is asymmetric payoffs. A model that correctly identifies large moves while making small errors on flat periods can generate positive risk-adjusted returns despite near-random accuracy. The 58–62% win rates in backtesting confirm this: the backtest engine's cost-aware execution filters out marginal signals.

### 6.3 Regime Effects

SPY regime analysis reveals that model performance varies significantly across market conditions. Bull regimes (45.5% of the period) naturally favor long-biased strategies. The framework's regime classification enables conditional strategy selection, though this paper does not implement dynamic regime switching.

### 6.4 Limitations

1. **Evaluation period**: ~2 years of hourly data limits statistical power for regime-conditional analysis
2. **Sentiment data**: PhraseBank provides a controlled evaluation but lacks temporal alignment with actual market events
3. **RL training**: 20,000 timesteps is relatively short for complex trading environments; longer training with curriculum learning may improve stability
4. **Long-only constraint**: Restricting to long/flat positions limits alpha generation in bear markets
5. **Fixed position sizing**: 100 shares per trade does not adapt to volatility or conviction
6. **Single-asset evaluation**: Each model trades one symbol independently; portfolio-level optimization is not addressed

---

## 7. Conclusion

START demonstrates that a well-engineered pipeline with rigorous temporal validation can produce meaningful trading signals from hourly equity data. The key findings are:

1. **Simple ML models outperform complex ones** at hourly frequency with limited data — regularized linear models (ridge, logistic) achieve Sharpe ratios of 2.0+ averaged across 12 symbols
2. **Walk-forward validation is essential** — the ~51% accuracy reflects genuine out-of-sample performance, avoiding the inflated metrics common in financial ML research
3. **Cost-aware backtesting reveals true performance** — strategies that appear profitable before costs may underperform after realistic transaction cost modeling
4. **Ablation studies quantify component contributions** — ML adds measurable alpha over technical baselines, while sentiment and RL contributions depend on data quality and training budget
5. **Reproducibility through engineering** — centralized configuration (78 parameters in settings.yaml), Parquet storage, and single-command execution ensure results can be replicated

The framework is designed for extensibility: adding new data sources requires implementing one provider class, new models integrate through a consistent fit/predict interface, and the dashboard automatically surfaces new results.

---

## 8. References

Brock, W., Lakonishok, J., & LeBaron, B. (1992). Simple technical trading rules and the stochastic properties of stock returns. *The Journal of Finance*, 47(5), 1731–1764.

Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research*, 270(2), 654–669.

Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *The Review of Financial Studies*, 33(5), 2223–2273.

Lo, A. W., Mamaysky, H., & Wang, J. (2000). Foundations of technical analysis: Computational algorithms, statistical inference, and empirical implementation. *The Journal of Finance*, 55(4), 1705–1765.

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). Good debt or bad debt: Detecting semantic orientations in economic texts. *Journal of the Association for Information Science and Technology*, 65(4), 782–796.

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529–533.

Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

---

## Appendix A: System Architecture

```
run_pipeline.py (orchestrator)
│
├── 01_data_ingestion.py      → data/raw/ → data/parquet/
├── 02_feature_engineering.py  → data/features/
├── 03_model_training.py       → data/results/model_comparison.parquet
├── 04_reinforcement_learning.py → data/models/*.zip, data/results/rl_comparison.parquet
├── 05_sentiment_analysis.py   → data/sentiment/, data/results/sentiment_scores.parquet
├── 06_ablation_backtest.py    → data/results/ablation_results.parquet
└── 07_dashboard_launcher.py   → Streamlit app (reads all results)
```

## Appendix B: Configuration Parameters

All 78 tunable parameters are centralized in `config/settings.yaml`. Key parameters:

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
| DQN | learning_rate | 1×10⁻⁴ |
| Backtest | slippage | 0.01% |
| Backtest | commission | $0.005/share |
| Backtest | initial_capital | $100,000 |
| Features | corr_threshold | 0.95 |

## Appendix C: Test Suite

26 unit tests across 4 test files verify core functionality:

| Test File | Tests | Coverage |
|-----------|:-----:|----------|
| test_clean.py | 4 | Data cleaning, dedup, empty handling |
| test_features.py | 5 | Indicators, returns, leakage prevention, correlation filter |
| test_backtest.py | 7 | Engine, metrics, drawdown, edge cases |
| test_env.py | 10 | RL environment, reward functions, episode termination |
