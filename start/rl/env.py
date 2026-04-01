"""
Custom Gymnasium trading environment for RL agents.

Observation: window of normalized features.
Actions: 0=Hold, 1=Buy, 2=Sell (long-only, no shorts).
Reward: After-cost PnL with drawdown penalty.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from start.utils.constants import SLIPPAGE_PCT, COMMISSION_PER_SHARE
from start.rl.rewards import shaped_reward
from start.utils.logger import get_logger

logger = get_logger(__name__)


class TradingEnv(gym.Env):
    """
    Long-only trading environment.

    Actions:
        0 = Hold (no change)
        1 = Buy (go long if flat)
        2 = Sell (go flat if long)

    Observation:
        Sliding window of normalized features + position indicator.

    Reward:
        After-cost return with drawdown penalty.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        window_size: int = 20,
        initial_capital: float = 100_000.0,
        shares_per_trade: int = 100,
        slippage_pct: float = SLIPPAGE_PCT,
        commission_per_share: float = COMMISSION_PER_SHARE,
    ):
        """
        Args:
            features: 2D array (n_bars, n_features) of normalized features.
            prices: 1D array of close prices aligned with features.
            window_size: Lookback window for observations.
            initial_capital: Starting capital.
            shares_per_trade: Fixed position size.
            slippage_pct: Slippage fraction per trade.
            commission_per_share: Commission per share per trade.
        """
        super().__init__()

        assert len(features) == len(prices), "Features and prices must align"
        assert len(features) > window_size, "Need more bars than window size"

        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float64)
        self.window_size = window_size
        self.initial_capital = initial_capital
        self.shares_per_trade = shares_per_trade
        self.slippage_pct = slippage_pct
        self.commission_per_share = commission_per_share

        self.n_features = features.shape[1]

        # Action space: Hold=0, Buy=1, Sell=2
        self.action_space = spaces.Discrete(3)

        # Observation: flattened window of features + position flag
        obs_dim = self.window_size * self.n_features + 1  # +1 for position
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # State
        self._current_step = 0
        self._position = 0  # 0=flat, 1=long
        self._cash = initial_capital
        self._shares_held = 0
        self._entry_price = 0.0
        self._peak_equity = initial_capital
        self._total_reward = 0.0

    def _get_obs(self) -> np.ndarray:
        """Build observation: flattened feature window + position."""
        start = max(0, self._current_step - self.window_size)
        end = self._current_step

        window = self.features[start:end]

        # Pad if at start
        if len(window) < self.window_size:
            pad = np.zeros((self.window_size - len(window), self.n_features), dtype=np.float32)
            window = np.vstack([pad, window])

        flat = window.flatten()
        obs = np.append(flat, [self._position]).astype(np.float32)
        return obs

    def _get_equity(self) -> float:
        """Current mark-to-market equity."""
        price = self.prices[self._current_step]
        return self._cash + self._shares_held * price

    def reset(self, seed=None, options=None):
        """Reset environment to start of episode."""
        super().reset(seed=seed)

        self._current_step = self.window_size  # Start after first window
        self._position = 0
        self._cash = self.initial_capital
        self._shares_held = 0
        self._entry_price = 0.0
        self._peak_equity = self.initial_capital
        self._total_reward = 0.0

        return self._get_obs(), {}

    def step(self, action: int):
        """
        Execute one step.

        Args:
            action: 0=Hold, 1=Buy, 2=Sell.

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        price_prev = self.prices[self._current_step - 1]
        price_now = self.prices[self._current_step]

        # Execute action
        old_position = self._position

        if action == 1 and self._position == 0:  # Buy
            buy_price = price_now * (1 + self.slippage_pct)
            cost = self.commission_per_share * self.shares_per_trade
            total_cost = buy_price * self.shares_per_trade + cost

            if self._cash >= total_cost:
                self._cash -= total_cost
                self._shares_held = self.shares_per_trade
                self._entry_price = buy_price
                self._position = 1

        elif action == 2 and self._position == 1:  # Sell
            sell_price = price_now * (1 - self.slippage_pct)
            cost = self.commission_per_share * self._shares_held
            proceeds = sell_price * self._shares_held - cost
            self._cash += proceeds
            self._shares_held = 0
            self._entry_price = 0.0
            self._position = 0

        # Compute reward
        equity = self._get_equity()
        self._peak_equity = max(self._peak_equity, equity)

        reward = shaped_reward(
            price_now=price_now,
            price_prev=price_prev,
            position=old_position,
            action=action,
            equity=equity,
            peak_equity=self._peak_equity,
            shares=self.shares_per_trade,
        )

        self._total_reward += reward
        self._current_step += 1

        # Episode termination
        terminated = self._current_step >= len(self.prices) - 1
        truncated = False

        # Force close position at end
        if terminated and self._position == 1:
            sell_price = self.prices[self._current_step] * (1 - self.slippage_pct)
            cost = self.commission_per_share * self._shares_held
            self._cash += sell_price * self._shares_held - cost
            self._shares_held = 0
            self._position = 0

        info = {
            "equity": equity,
            "position": self._position,
            "total_reward": self._total_reward,
            "step": self._current_step,
        }

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)

        return obs, reward, terminated, truncated, info

    def render(self):
        """Print current state."""
        equity = self._get_equity()
        ret = (equity - self.initial_capital) / self.initial_capital
        print(
            f"Step {self._current_step}/{len(self.prices)} | "
            f"Pos: {'LONG' if self._position else 'FLAT'} | "
            f"Equity: ${equity:,.2f} ({ret:+.2%}) | "
            f"Peak: ${self._peak_equity:,.2f}"
        )
