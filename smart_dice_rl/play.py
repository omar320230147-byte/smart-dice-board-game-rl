from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .environment import SmartDiceEnv
from .tabular import TabularAgent
from .sb3_agents import ALGORITHMS

ROOT = Path(__file__).resolve().parent.parent

def play(env, policy, label, kind, seed=2026):
    obs, _ = env.reset(seed=seed); state = int(obs[0]); rng = np.random.default_rng(seed)
    total, steps, done = 0., 0, False
    print(f"\n{label} | start={state}, goal={env.size}")
    while not done:
        if kind == "random": action = int(rng.integers(6))
        elif kind == "tabular": action = policy.act(state, rng, False)
        else:
            action, _ = policy.predict(obs, deterministic=True); action = int(np.asarray(action).item())
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1; total += reward; state = int(obs[0]); done = terminated or truncated
        print(f"move {steps:3d}: {info['old_position']:3d} --dice {info['dice']}--> {info['position']:3d}  event={info['event'] or '-'} reward={reward:g}")
    print(f"result: {'WON' if terminated else 'LIMIT'} in {steps} moves; return={total:g}")

def main():
    p=argparse.ArgumentParser(description="Show random and learned Smart Dice agents")
    p.add_argument("--board",default="board_1_30"); p.add_argument("--algorithm",default="DQN",choices=["DQN","PPO","A2C","q_learning","sarsa"]); p.add_argument("--seed",type=int,default=17)
    args=p.parse_args(); board_file = "_".join(args.board.split("_")[1:])
    board=json.loads((ROOT/"data"/"generated_boards"/f"board_{board_file}.json").read_text(encoding="utf-8"))
    board["snakes"] = {int(k): int(v) for k, v in board["snakes"].items()}
    board["ladders"] = {int(k): int(v) for k, v in board["ladders"].items()}
    env=SmartDiceEnv(board["size"],board["snakes"],board["ladders"])
    play(env,None,"Untrained random policy","random")
    if args.algorithm in ("q_learning","sarsa"):
        path=ROOT/"models"/f"{args.board}_{args.algorithm}_seed{args.seed}.npz"
        agent=TabularAgent(board["size"],args.algorithm); agent.q=np.load(path)["q_table"]; kind="tabular"
    else:
        path=ROOT/"models"/f"{args.board}_{args.algorithm.lower()}_seed{args.seed}.zip"
        agent=ALGORITHMS[args.algorithm].load(str(path),device="cpu"); kind="sb3"
    play(env,agent,f"Trained {args.algorithm}",kind)

if __name__ == "__main__": main()
