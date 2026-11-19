from __future__ import annotations
import numpy as np


class TabularAgent:
    def __init__(self, size: int, method: str, alpha=.15, gamma=.99, epsilon=1.0, epsilon_min=.03, epsilon_decay=.999):
        if method not in ("q_learning", "sarsa"):
            raise ValueError(method)
        self.q = np.zeros((size + 1, 6), dtype=np.float32)
        self.method, self.alpha, self.gamma = method, alpha, gamma
        self.epsilon, self.epsilon_min, self.epsilon_decay = epsilon, epsilon_min, epsilon_decay

    def act(self, state: int, rng: np.random.Generator, explore=False) -> int:
        if explore and rng.random() < self.epsilon:
            return int(rng.integers(6))
        return int(np.argmax(self.q[state]))

    def train(self, env, episodes: int, seed: int, progress_every=100, on_checkpoint=None):
        rng = np.random.default_rng(seed)
        records, recent = [], []
        for ep in range(1, episodes + 1):
            obs, _ = env.reset(seed=seed + ep)
            state = int(obs[0]); action = self.act(state, rng, True)
            total, moves, done = 0.0, 0, False
            while not done:
                nxt, reward, terminated, truncated, _ = env.step(action)
                ns, done = int(nxt[0]), terminated or truncated
                next_action = self.act(ns, rng, True) if not done else 0
                if self.method == "q_learning":
                    bootstrap = float(np.max(self.q[ns])) if not done else 0.0
                else:
                    bootstrap = float(self.q[ns, next_action]) if not done else 0.0
                self.q[state, action] += self.alpha * (reward + self.gamma * bootstrap - self.q[state, action])
                total += reward; moves += 1; state = ns; action = next_action
            recent.append((total, moves, bool(terminated)))
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if ep % progress_every == 0 or ep == episodes:
                arr = np.asarray(recent[-progress_every:])
                record = {"episode": ep, "avg_reward": float(arr[:, 0].mean()), "avg_moves": float(arr[:, 1].mean()), "success_rate": float(arr[:, 2].mean()), "epsilon": self.epsilon}
                if on_checkpoint is not None and ep % 500 == 0:
                    record.update(on_checkpoint(ep, self))
                records.append(record)
        return records

