from __future__ import annotations
from pathlib import Path
import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback

ALGORITHMS = {"DQN": DQN, "PPO": PPO, "A2C": A2C}


def create_model(name, env, seed):
    cls = ALGORITHMS[name]
    common = dict(policy="MlpPolicy", env=env, seed=seed, device="cpu", policy_kwargs={"net_arch": [64, 64]}, verbose=0)
    if name == "DQN":
        return cls(**common, learning_rate=1e-3, buffer_size=20_000, learning_starts=500, batch_size=64, gamma=.99, train_freq=4, target_update_interval=500, exploration_fraction=.5, exploration_final_eps=.05)
    if name == "PPO":
        return cls(**common, learning_rate=3e-4, n_steps=256, batch_size=64, n_epochs=5, gamma=.99, gae_lambda=.95, ent_coef=.01)
    return cls(**common, learning_rate=7e-4, n_steps=16, gamma=.99, gae_lambda=.95, ent_coef=.01)


class PeriodicEvalCallback(BaseCallback):
    def __init__(self, make_eval_env, every: int, episodes: int, output: Path):
        super().__init__(verbose=0)
        self.make_eval_env, self.every, self.episodes, self.output = make_eval_env, every, episodes, output
        self.rows, self.last_eval, self.high_windows = [], 0, 0

    def _on_step(self):
        if self.num_timesteps - self.last_eval < self.every:
            return True
        self.last_eval = self.num_timesteps
        env = self.make_eval_env()
        rewards, lengths, wins = [], [], []
        for i in range(self.episodes):
            obs, _ = env.reset(seed=91000 + i)
            done = False; reward_sum = 0.; steps = 0; success = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
                reward_sum += reward; steps += 1; done = terminated or truncated; success = bool(info["success"])
            rewards.append(reward_sum); lengths.append(steps); wins.append(success)
        env.close()
        success_rate = float(np.mean(wins))
        self.high_windows = self.high_windows + 1 if success_rate >= .95 else 0
        self.rows.append({"episode": self.num_timesteps, "avg_reward": float(np.mean(rewards)), "avg_moves": float(np.mean(lengths)), "success_rate": success_rate})
        return True

