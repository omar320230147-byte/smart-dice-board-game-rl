"""Board-aware fixed-width observations and randomized-board training envs."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .environment import SmartDiceEnv

MAX_BOARD_SIZE = 100
OBS_SIZE = 2 + 2 * MAX_BOARD_SIZE


def encode_board_state(position: int, board: dict, max_size: int = MAX_BOARD_SIZE) -> np.ndarray:
    """Encode normalized position, board size, and scaled snake/ladder maps."""
    size = int(board["size"])
    if size > max_size:
        raise ValueError(f"board size {size} exceeds encoder limit {max_size}")
    out = np.zeros(2 + 2 * max_size, dtype=np.float32)
    out[0] = (int(position) - 1) / (size - 1)
    out[1] = size / max_size
    for key, value in board["snakes"].items():
        idx = int(round((int(key) - 1) * (max_size - 1) / (size - 1)))
        out[2 + idx] = max(out[2 + idx], (int(value) - int(key)) / size)
    for key, value in board["ladders"].items():
        idx = int(round((int(key) - 1) * (max_size - 1) / (size - 1)))
        out[2 + max_size + idx] = max(out[2 + max_size + idx], (int(value) - int(key)) / size)
    return out


class BoardCollectionEnv(gym.Env):
    """Sample one board at each reset; board identity is observable in every state."""
    metadata = {"render_modes": []}

    def __init__(self, boards: list[dict], max_steps_factor: int = 4):
        super().__init__()
        if not boards:
            raise ValueError("at least one board is required")
        self.boards = [self._normalize(b) for b in boards]
        if any(int(b["size"]) > MAX_BOARD_SIZE for b in self.boards):
            raise ValueError(f"all boards must be at most {MAX_BOARD_SIZE} squares")
        self.max_steps_factor = max_steps_factor
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_SIZE,), dtype=np.float32)
        self.board = self.boards[0]
        self.env = self._make_env(self.board)

    @staticmethod
    def _normalize(board):
        return {"size": int(board["size"]),
                "snakes": {int(k): int(v) for k, v in board["snakes"].items()},
                "ladders": {int(k): int(v) for k, v in board["ladders"].items()},
                "seed": board.get("seed")}

    def _make_env(self, board):
        return SmartDiceEnv(board["size"], board["snakes"], board["ladders"],
                            max_steps=board["size"] * self.max_steps_factor)

    def _encode(self, position):
        return encode_board_state(position, self.board)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        ix = int(self.np_random.integers(len(self.boards)))
        self.board = self.boards[ix]
        self.env = self._make_env(self.board)
        child_seed = int(self.np_random.integers(0, 2**31 - 1))
        _, info = self.env.reset(seed=child_seed, options=options)
        info["board_seed"] = self.board.get("seed")
        return self._encode(1), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["board_seed"] = self.board.get("seed")
        return self._encode(int(obs[0])), reward, terminated, truncated, info


def fixed_board_env(board: dict):
    return BoardCollectionEnv([board])


def benchmark_inference_ms(model, obs, repeats: int = 1000) -> float:
    """Return mean CPU prediction latency for one observation (not batched)."""
    import time
    start=time.perf_counter()
    for _ in range(repeats):
        model.predict(obs, deterministic=True)
    return (time.perf_counter()-start)*1000/repeats


def load_board_file(path):
    import json
    return json.loads(path.read_text(encoding="utf-8"))
