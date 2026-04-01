"""
RL agent wrappers using stable-baselines3.

Provides PPO and DQN agents for the TradingEnv.
"""

from typing import Optional
from pathlib import Path

import numpy as np
import pandas as pd

from start.rl.env import TradingEnv
from start.features.builder import get_feature_columns
from start.utils.logger import get_logger

logger = get_logger(__name__)


def _get_rl_config():
    """Load RL config from settings."""
    from config import get_setting
    return get_setting("rl", default={})


def _prepare_env_data(df: pd.DataFrame, feature_cols: Optional[list] = None):
    """Extract normalized features and prices from DataFrame."""
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    features = df[feature_cols].values.astype(np.float32)
    prices = df["close"].values.astype(np.float64)

    # Normalize features (z-score)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std == 0] = 1.0
    features = (features - mean) / std

    return features, prices, mean, std


def train_ppo(
    df: pd.DataFrame,
    total_timesteps: int = None,
    learning_rate: float = None,
    window_size: int = None,
    feature_cols: Optional[list] = None,
    save_path: Optional[str] = None,
) -> dict:
    """
    Train a PPO agent on trading data.

    Args:
        df: Feature DataFrame for a single symbol.
        total_timesteps: Total training steps.
        learning_rate: PPO learning rate.
        window_size: Observation window size.
        feature_cols: Feature columns to use.
        save_path: Optional path to save trained model.

    Returns:
        Dict with model, env info, and training stats.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import EvalCallback

    rl_cfg = _get_rl_config()
    ppo_cfg = rl_cfg.get("ppo", {})
    total_timesteps = total_timesteps or rl_cfg.get("total_timesteps", 50000)
    learning_rate = learning_rate or ppo_cfg.get("learning_rate", 3e-4)
    window_size = window_size or rl_cfg.get("window_size", 20)
    split_ratio = rl_cfg.get("train_eval_split", 0.8)

    features, prices, mean, std = _prepare_env_data(df, feature_cols)

    # Split for train/eval
    split = int(len(features) * split_ratio)

    train_env = TradingEnv(
        features=features[:split],
        prices=prices[:split],
        window_size=window_size,
    )

    eval_env = TradingEnv(
        features=features[split:],
        prices=prices[split:],
        window_size=window_size,
    )

    logger.info(f"[PPO] Training: {split} bars, Eval: {len(features) - split} bars")
    logger.info(f"[PPO] Features: {features.shape[1]}, Timesteps: {total_timesteps}")

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=ppo_cfg.get("n_steps", 256),
        batch_size=ppo_cfg.get("batch_size", 64),
        n_epochs=ppo_cfg.get("n_epochs", 10),
        gamma=ppo_cfg.get("gamma", 0.99),
        gae_lambda=ppo_cfg.get("gae_lambda", 0.95),
        clip_range=ppo_cfg.get("clip_range", 0.2),
        verbose=0,
    )

    # Train
    model.learn(total_timesteps=total_timesteps)

    # Save
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info(f"[PPO] Model saved to {save_path}")

    # Evaluate on eval set
    eval_metrics = _evaluate_agent(model, eval_env)

    return {
        "model": model,
        "agent_type": "PPO",
        "train_bars": split,
        "eval_bars": len(features) - split,
        "total_timesteps": total_timesteps,
        "eval_metrics": eval_metrics,
        "feature_mean": mean,
        "feature_std": std,
    }


def train_dqn(
    df: pd.DataFrame,
    total_timesteps: int = None,
    learning_rate: float = None,
    window_size: int = None,
    feature_cols: Optional[list] = None,
    save_path: Optional[str] = None,
) -> dict:
    """
    Train a DQN agent on trading data.

    Args:
        df: Feature DataFrame for a single symbol.
        total_timesteps: Total training steps.
        learning_rate: DQN learning rate.
        window_size: Observation window size.
        feature_cols: Feature columns to use.
        save_path: Optional path to save trained model.

    Returns:
        Dict with model, env info, and training stats.
    """
    from stable_baselines3 import DQN

    rl_cfg = _get_rl_config()
    dqn_cfg = rl_cfg.get("dqn", {})
    total_timesteps = total_timesteps or rl_cfg.get("total_timesteps", 50000)
    learning_rate = learning_rate or dqn_cfg.get("learning_rate", 1e-4)
    window_size = window_size or rl_cfg.get("window_size", 20)
    split_ratio = rl_cfg.get("train_eval_split", 0.8)

    features, prices, mean, std = _prepare_env_data(df, feature_cols)

    split = int(len(features) * split_ratio)

    train_env = TradingEnv(
        features=features[:split],
        prices=prices[:split],
        window_size=window_size,
    )

    eval_env = TradingEnv(
        features=features[split:],
        prices=prices[split:],
        window_size=window_size,
    )

    logger.info(f"[DQN] Training: {split} bars, Eval: {len(features) - split} bars")
    logger.info(f"[DQN] Features: {features.shape[1]}, Timesteps: {total_timesteps}")

    model = DQN(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        buffer_size=dqn_cfg.get("buffer_size", 10000),
        learning_starts=dqn_cfg.get("learning_starts", 500),
        batch_size=dqn_cfg.get("batch_size", 64),
        gamma=dqn_cfg.get("gamma", 0.99),
        target_update_interval=dqn_cfg.get("target_update_interval", 250),
        exploration_fraction=dqn_cfg.get("exploration_fraction", 0.3),
        exploration_final_eps=dqn_cfg.get("exploration_final_eps", 0.05),
        verbose=0,
    )

    model.learn(total_timesteps=total_timesteps)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        logger.info(f"[DQN] Model saved to {save_path}")

    eval_metrics = _evaluate_agent(model, eval_env)

    return {
        "model": model,
        "agent_type": "DQN",
        "train_bars": split,
        "eval_bars": len(features) - split,
        "total_timesteps": total_timesteps,
        "eval_metrics": eval_metrics,
        "feature_mean": mean,
        "feature_std": std,
    }


def _evaluate_agent(model, env: TradingEnv, n_episodes: int = 1) -> dict:
    """
    Evaluate a trained RL agent on an environment.

    Returns:
        Dict with episode metrics.
    """
    all_rewards = []
    all_equities = []
    all_actions = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0.0
        episode_actions = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += reward
            episode_actions.append(int(action))
            done = terminated or truncated

        final_equity = info.get("equity", env.initial_capital)
        all_rewards.append(episode_reward)
        all_equities.append(final_equity)
        all_actions.extend(episode_actions)

    # Action distribution
    actions = np.array(all_actions)
    n_holds = (actions == 0).sum()
    n_buys = (actions == 1).sum()
    n_sells = (actions == 2).sum()
    total_actions = len(actions)

    final_equity = np.mean(all_equities)
    total_return = (final_equity - env.initial_capital) / env.initial_capital

    metrics = {
        "mean_reward": np.mean(all_rewards),
        "final_equity": final_equity,
        "total_return": total_return,
        "n_holds": int(n_holds),
        "n_buys": int(n_buys),
        "n_sells": int(n_sells),
        "hold_pct": n_holds / total_actions if total_actions > 0 else 0,
        "buy_pct": n_buys / total_actions if total_actions > 0 else 0,
        "sell_pct": n_sells / total_actions if total_actions > 0 else 0,
    }

    logger.info(
        f"  Eval: equity=${final_equity:,.2f} ({total_return:+.2%}), "
        f"actions: hold={n_holds} buy={n_buys} sell={n_sells}"
    )

    return metrics


def generate_rl_signals(
    model,
    df: pd.DataFrame,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    feature_cols: Optional[list] = None,
    window_size: int = 20,
) -> pd.Series:
    """
    Generate trading signals from a trained RL agent.

    Args:
        model: Trained stable-baselines3 model.
        df: Feature DataFrame.
        feature_mean: Feature means used during training.
        feature_std: Feature stds used during training.
        feature_cols: Feature columns.
        window_size: Observation window size.

    Returns:
        Series of signals (0=flat, 1=long) aligned with df index.
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    features = df[feature_cols].values.astype(np.float32)
    prices = df["close"].values.astype(np.float64)

    # Normalize with training stats
    features = (features - feature_mean) / feature_std

    env = TradingEnv(
        features=features,
        prices=prices,
        window_size=window_size,
    )

    obs, _ = env.reset()
    signals = np.zeros(len(df), dtype=int)
    position = 0

    for i in range(window_size, len(df) - 1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))

        if int(action) == 1 and position == 0:
            position = 1
        elif int(action) == 2 and position == 1:
            position = 0

        signals[i] = position

        if terminated or truncated:
            break

    return pd.Series(signals, index=df.index, name="signal")
