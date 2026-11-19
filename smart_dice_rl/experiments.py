from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env as sb3_check_env
from gymnasium.utils.env_checker import check_env

from .boards import save_required_boards
from .environment import SmartDiceEnv
from .tabular import TabularAgent
from .sb3_agents import create_model, PeriodicEvalCallback, ALGORITHMS
from .baselines import bfs_distance

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
MODELS = ROOT / "models"
SEEDS = (17, 29, 41)
TABULAR_EPISODES = 3500
SB3_TIMESTEPS = 6000
EVAL_EPISODES = 100


def make_env(board, max_steps=None):
    return SmartDiceEnv(board["size"], board["snakes"], board["ladders"], max_steps=max_steps)


def evaluate(agent, board, episodes=EVAL_EPISODES, seed=70000, kind="tabular"):
    env = make_env(board); rng = np.random.default_rng(seed)
    rewards, lengths, successes = [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep); state = int(obs[0]); total = 0.; steps = 0; done = False; success = False
        while not done:
            if kind == "random": action = int(rng.integers(6))
            elif kind == "sb3":
                action, _ = agent.predict(obs, deterministic=True); action = int(np.asarray(action).item())
            else: action = agent.act(state, rng, explore=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward; steps += 1; state = int(obs[0]); done = terminated or truncated; success = bool(info["success"])
        rewards.append(total); lengths.append(steps); successes.append(success)
    env.close()
    return {"success_rate": float(np.mean(successes)), "avg_moves": float(np.mean(lengths)), "avg_reward": float(np.mean(rewards)), "eval_episodes": episodes}


def run():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    (OUT / "csv").mkdir(parents=True, exist_ok=True); (OUT / "plots").mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    boards = save_required_boards(ROOT / "data" / "generated_boards")
    (ROOT / "config").mkdir(exist_ok=True)
    (ROOT / "config" / "experiments.json").write_text(json.dumps({"boards": boards, "seeds": SEEDS, "tabular_episodes": TABULAR_EPISODES, "sb3_timesteps": SB3_TIMESTEPS, "evaluation_episodes": EVAL_EPISODES, "convergence": "three consecutive evaluation windows with >=95% success"}, indent=2), encoding="utf-8")
    # Both required checkers run against the first generated configuration.
    e = make_env(boards[0]); check_env(e, skip_render_check=True); sb3_check_env(e, warn=True); e.close()
    summaries, progress, bfs_rows = [], [], []
    for bi, board in enumerate(boards, 1):
        board_id = f"board_{bi}_{board['size']}"
        bfs_rows.append({"board": board_id, "board_size": board["size"], "bfs_min_moves": bfs_distance(make_env(board))})
        rand = evaluate(None, board, kind="random", seed=61000 + bi)
        summaries.append({"board": board_id, "board_size": board["size"], "snakes": len(board["snakes"]), "ladders": len(board["ladders"]), "algorithm": "Random", "seed": 0, "training_budget": 0, "training_time_seconds": 0., **rand, "convergence_episode": "N/A"})
        for seed in SEEDS:
            for algo in ("q_learning", "sarsa"):
                env = make_env(board); agent = TabularAgent(board["size"], algo)
                started = time.perf_counter()
                rows = agent.train(env, TABULAR_EPISODES, seed, on_checkpoint=lambda ep, a, b=board, s=seed: {"eval_success_rate": evaluate(a, b, episodes=100, seed=81000 + s, kind="tabular")["success_rate"]})
                elapsed = time.perf_counter() - started; env.close()
                model_path = MODELS / f"{board_id}_{algo}_seed{seed}.npz"; np.savez_compressed(model_path, q_table=agent.q)
                ev = evaluate(agent, board, seed=70000 + seed)
                for row in rows: progress.append({"board": board_id, "algorithm": algo, "seed": seed, **row})
                eval_rows = [r for r in rows if "eval_success_rate" in r]
                convergence = next((eval_rows[i]["episode"] for i in range(2, len(eval_rows)) if all(x["eval_success_rate"] >= .95 for x in eval_rows[i-2:i+1])), "Not converged within training budget")
                summaries.append({"board": board_id, "board_size": board["size"], "snakes": len(board["snakes"]), "ladders": len(board["ladders"]), "algorithm": algo, "seed": seed, "training_budget": TABULAR_EPISODES, "training_time_seconds": elapsed, **ev, "convergence_episode": convergence})
        for seed in SEEDS:
            for algo in ("DQN", "PPO", "A2C"):
                env = Monitor(make_env(board))
                model = create_model(algo, env, seed)
                callback = PeriodicEvalCallback(lambda b=board: make_env(b), every=max(1000, SB3_TIMESTEPS // 8), episodes=50, output=OUT)
                started = time.perf_counter(); model.learn(total_timesteps=SB3_TIMESTEPS, callback=callback, progress_bar=False); elapsed = time.perf_counter() - started
                path = MODELS / f"{board_id}_{algo.lower()}_seed{seed}"
                model.save(str(path)); ev = evaluate(model, board, seed=70000 + seed, kind="sb3")
                for row in callback.rows: progress.append({"board": board_id, "algorithm": algo, "seed": seed, **row})
                convergence = next((r["episode"] for i, r in enumerate(callback.rows) if i >= 2 and all(x["success_rate"] >= .95 for x in callback.rows[i-2:i+1])), "Not converged within training budget")
                summaries.append({"board": board_id, "board_size": board["size"], "snakes": len(board["snakes"]), "ladders": len(board["ladders"]), "algorithm": algo, "seed": seed, "training_budget": SB3_TIMESTEPS, "training_time_seconds": elapsed, **ev, "convergence_episode": convergence})
                env.close()
                print(f"finished {board_id} {algo} seed={seed} in {elapsed:.1f}s; success={ev['success_rate']:.3f}", flush=True)
        print(f"finished tabular runs for {board_id}", flush=True)
    pd.DataFrame(summaries).to_csv(OUT / "csv" / "evaluation_summary.csv", index=False)
    pd.DataFrame(progress).to_csv(OUT / "csv" / "training_progress.csv", index=False)
    pd.DataFrame(bfs_rows).to_csv(OUT / "csv" / "bfs_baseline.csv", index=False)
    from .plots import make_plots
    make_plots(OUT)


def refresh_tabular_results():
    """Re-run tabular checkpoints with separate convergence evaluations."""
    boards = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data" / "generated_boards").glob("board_*.json"))]
    for board in boards:
        board["snakes"] = {int(k): int(v) for k, v in board["snakes"].items()}
        board["ladders"] = {int(k): int(v) for k, v in board["ladders"].items()}
    summary_path, progress_path = OUT / "csv" / "evaluation_summary.csv", OUT / "csv" / "training_progress.csv"
    summary = pd.read_csv(summary_path).to_dict("records")
    summary = [r for r in summary if r["algorithm"] not in ("q_learning", "sarsa")]
    progress = pd.read_csv(progress_path).to_dict("records")
    progress = [r for r in progress if r["algorithm"] not in ("q_learning", "sarsa")]
    for bi, board in enumerate(boards, 1):
        board_id = f"board_{bi}_{board['size']}"
        for seed in SEEDS:
            for algo in ("q_learning", "sarsa"):
                env = make_env(board); agent = TabularAgent(board["size"], algo); started = time.perf_counter()
                rows = agent.train(env, TABULAR_EPISODES, seed, on_checkpoint=lambda ep, a, b=board, s=seed: {"eval_success_rate": evaluate(a, b, episodes=100, seed=81000 + s, kind="tabular")["success_rate"]})
                elapsed = time.perf_counter() - started; env.close()
                np.savez_compressed(MODELS / f"{board_id}_{algo}_seed{seed}.npz", q_table=agent.q)
                ev = evaluate(agent, board, seed=70000 + seed)
                progress.extend({"board": board_id, "algorithm": algo, "seed": seed, **row} for row in rows)
                eval_rows = [r for r in rows if "eval_success_rate" in r]
                convergence = next((eval_rows[i]["episode"] for i in range(2, len(eval_rows)) if all(x["eval_success_rate"] >= .95 for x in eval_rows[i-2:i+1])), "Not converged within training budget")
                summary.append({"board": board_id, "board_size": board["size"], "snakes": len(board["snakes"]), "ladders": len(board["ladders"]), "algorithm": algo, "seed": seed, "training_budget": TABULAR_EPISODES, "training_time_seconds": elapsed, **ev, "convergence_episode": convergence})
                print(f"refreshed {board_id} {algo} seed={seed} in {elapsed:.1f}s; success={ev['success_rate']:.3f}", flush=True)
    pd.DataFrame(summary).to_csv(summary_path, index=False); pd.DataFrame(progress).to_csv(progress_path, index=False)
    from .plots import make_plots
    make_plots(OUT)


def evaluate_saved_models():
    """Re-evaluate saved learned and random agents on held-out episodes."""
    boards = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((ROOT / "data" / "generated_boards").glob("board_*.json"))]
    for board in boards:
        board["snakes"] = {int(k): int(v) for k, v in board["snakes"].items()}
        board["ladders"] = {int(k): int(v) for k, v in board["ladders"].items()}
    out = []
    for bi, board in enumerate(boards, 1):
        board_id = f"board_{bi}_{board['size']}"
        out.append({"board": board_id, "board_size": board["size"], "snakes": len(board["snakes"]), "ladders": len(board["ladders"]), "algorithm": "Random", "seed": 0, **evaluate(None, board, kind="random", seed=61000 + bi)})
        for seed in SEEDS:
            for algo in ("q_learning", "sarsa"):
                agent = TabularAgent(board["size"], algo)
                agent.q = np.load(MODELS / f"{board_id}_{algo}_seed{seed}.npz")["q_table"]
                out.append({"board": board_id, "board_size": board["size"], "snakes": len(board["snakes"]), "ladders": len(board["ladders"]), "algorithm": algo, "seed": seed, **evaluate(agent, board, seed=70000 + seed)})
            for algo in ("DQN", "PPO", "A2C"):
                model = ALGORITHMS[algo].load(str(MODELS / f"{board_id}_{algo.lower()}_seed{seed}.zip"), device="cpu")
                out.append({"board": board_id, "board_size": board["size"], "snakes": len(board["snakes"]), "ladders": len(board["ladders"]), "algorithm": algo, "seed": seed, **evaluate(model, board, seed=70000 + seed, kind="sb3")})
    pd.DataFrame(out).to_csv(OUT / "csv" / "evaluation_rerun.csv", index=False)
    print(f"Evaluated saved policies; wrote {OUT / 'csv' / 'evaluation_rerun.csv'}")

