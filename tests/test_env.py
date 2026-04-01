"""Tests for RL trading environment."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from start.rl.env import TradingEnv
from start.rl.rewards import after_cost_reward, drawdown_penalty, shaped_reward


def _make_env(n=500, n_features=5, window_size=20):
    """Create a TradingEnv with synthetic data."""
    np.random.seed(42)
    features = np.random.randn(n, n_features).astype(np.float32)
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    prices = prices.astype(np.float64)
    return TradingEnv(features=features, prices=prices, window_size=window_size)


def test_env_reset():
    env = _make_env()
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert isinstance(info, dict)


def test_env_step():
    env = _make_env()
    obs, _ = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # Hold
    assert obs.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert not terminated


def test_env_buy_sell_cycle():
    env = _make_env()
    env.reset()

    # Buy
    _, _, _, _, info1 = env.step(1)
    assert info1["position"] == 1

    # Hold
    env.step(0)

    # Sell
    _, _, _, _, info2 = env.step(2)
    assert info2["position"] == 0


def test_env_episode_terminates():
    env = _make_env(n=50, window_size=5)
    env.reset()
    done = False
    steps = 0
    while not done:
        _, _, terminated, truncated, _ = env.step(0)
        done = terminated or truncated
        steps += 1
    assert steps > 0
    assert steps <= 50


def test_env_action_space():
    env = _make_env()
    assert env.action_space.n == 3  # Hold, Buy, Sell


def test_reward_after_cost():
    r = after_cost_reward(
        price_now=101, price_prev=100,
        position=1, action=0, shares=100,
    )
    assert r > 0  # Price went up while long

    r2 = after_cost_reward(
        price_now=99, price_prev=100,
        position=1, action=0, shares=100,
    )
    assert r2 < 0  # Price went down while long


def test_reward_flat_position():
    r = after_cost_reward(
        price_now=101, price_prev=100,
        position=0, action=0, shares=100,
    )
    assert r == 0.0  # No PnL when flat


def test_drawdown_penalty_zero():
    p = drawdown_penalty(equity=100000, peak_equity=100000)
    assert p == 0.0  # No drawdown


def test_drawdown_penalty_negative():
    p = drawdown_penalty(equity=90000, peak_equity=100000)
    assert p < 0  # Should be penalized


def test_shaped_reward():
    r = shaped_reward(
        price_now=101, price_prev=100,
        position=1, action=0,
        equity=101000, peak_equity=101000,
        shares=100,
    )
    assert isinstance(r, float)
