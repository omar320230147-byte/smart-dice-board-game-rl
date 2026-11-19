from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .boards import validate_board


class SmartDiceEnv(gym.Env):
    """Deterministic board: action index 0..5 means dice value 1..6."""
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(self, size: int, snakes: dict[int, int], ladders: dict[int, int], max_steps: int | None = None, render_mode: str | None = None):
        super().__init__()
        validate_board(size, snakes, ladders)
        if render_mode not in (None, "human", "ansi"):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        self.size, self.snakes, self.ladders = int(size), dict(snakes), dict(ladders)
        self.max_steps = int(max_steps or size * 4)
        self.render_mode = render_mode
        self.observation_space = spaces.Box(1, size, shape=(1,), dtype=np.int32)
        self.action_space = spaces.Discrete(6)
        self.position, self.steps = 1, 0

    def _obs(self):
        return np.asarray([self.position], dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.position, self.steps = 1, 0
        return self._obs(), {"position": 1}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Action must be an integer in [0, 5], got {action!r}")
        old, dice = self.position, int(action) + 1
        target, event, reward = old + dice, None, -1.0
        self.steps += 1
        if target > self.size:
            target, reward, event = old, -10.0, "overshoot"
        elif target in self.snakes:
            target, event, reward = self.snakes[target], "snake", -20.0
        elif target in self.ladders:
            target, event, reward = self.ladders[target], "ladder", 20.0
        self.position = target
        terminated = self.position == self.size
        if terminated:
            reward = 100.0
        truncated = self.steps >= self.max_steps and not terminated
        info = {"old_position": old, "dice": dice, "target": old + dice, "position": target, "event": event, "success": terminated}
        return self._obs(), reward, terminated, truncated, info

    def render(self):
        text = f"Position {self.position}/{self.size} | snakes={self.snakes} | ladders={self.ladders}"
        if self.render_mode == "human":
            print(text)
            return None
        return text

