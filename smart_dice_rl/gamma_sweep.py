"""Ten-seed 100-square discount sweep with isolated final evaluation."""
from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pandas as pd

from .environment import SmartDiceEnv
from .experiments import evaluate
from .improve_100 import ROOT, load_board
from .tabular import TabularAgent

GAMMAS = (.80, .90, .95, .99)
SEEDS = (301, 307, 311, 313, 317, 331, 337, 347, 349, 353)
EPISODES = 5000


def run():
    board=load_board(); out=ROOT/"results"/"improved"/"gamma_sweep"
    models=ROOT/"models"/"improved"/"gamma_sweep_10seed"
    out.mkdir(parents=True,exist_ok=True); models.mkdir(parents=True,exist_ok=True)
    rows=[]; progress=[]
    for gamma in GAMMAS:
        for algo in ("q_learning","sarsa"):
            for seed in SEEDS:
                env=SmartDiceEnv(board["size"],board["snakes"],board["ladders"])
                agent=TabularAgent(board["size"],algo,alpha=.1,gamma=gamma,epsilon=1.,epsilon_min=.02,epsilon_decay=.9998)
                def checkpoint(ep, a, seed=seed):
                    metrics=evaluate(a,board,episodes=25,seed=810000+seed,kind="tabular")
                    return {"eval_success_rate":metrics["success_rate"],"eval_avg_moves":metrics["avg_moves"],"eval_avg_reward":metrics["avg_reward"]}
                start=time.perf_counter()
                points=agent.train(env,EPISODES,seed,progress_every=500,on_checkpoint=checkpoint)
                elapsed=time.perf_counter()-start; env.close()
                path=models/f"{algo}_gamma{gamma:.2f}_seed{seed}.npz"
                np.savez_compressed(path,q_table=agent.q)
                ev=evaluate(agent,board,episodes=100,seed=910000+seed,kind="tabular")
                eval_points=[p for p in points if "eval_success_rate" in p]
                conv=next((eval_points[i]["episode"] for i in range(2,len(eval_points))
                           if all(x["eval_success_rate"]>=.95 for x in eval_points[i-2:i+1])),None)
                rows.append({"board":"board_3_100","algorithm":algo,"gamma":gamma,"seed":seed,
                             "training_episodes":EPISODES,"training_time_seconds":elapsed,
                             "convergence_episode":conv if conv is not None else "Not converged within training budget",
                             "final_threshold_sustained":ev["success_rate"]>=.95,
                             "failure_rate":1-ev["success_rate"],"model_path":str(path.relative_to(ROOT)),**ev})
                progress.extend({"algorithm":algo,"gamma":gamma,"seed":seed,**p} for p in points)
                print(f"gamma={gamma:.2f} {algo} seed={seed}: success={ev['success_rate']:.2f}, moves={ev['avg_moves']:.1f}",flush=True)
    (out/"config.json").write_text(__import__("json").dumps({"board":"board_3_100","gammas":GAMMAS,"seeds":SEEDS,
        "episodes":EPISODES,"alpha":.1,"epsilon_start":1.0,"epsilon_min":.02,"epsilon_decay":.9998,
        "eval_episodes_final":100,"checkpoint_eval_episodes":25,"reward":"original"},indent=2),encoding="utf-8")
    pd.DataFrame(rows).to_csv(out/"gamma_sweep_summary.csv",index=False)
    pd.DataFrame(progress).to_csv(out/"gamma_sweep_progress.csv",index=False)
    summary=pd.DataFrame(rows).groupby(["algorithm","gamma"]).agg(
        seeds=("seed","count"),success_mean=("success_rate","mean"),success_sd=("success_rate","std"),
        moves_mean=("avg_moves","mean"),moves_sd=("avg_moves","std"),reward_mean=("avg_reward","mean"),
        reward_sd=("avg_reward","std"),time_mean=("training_time_seconds","mean"),failures=("failure_rate",lambda x:int((x>0).sum()))).reset_index()
    # Wilson interval for the fraction of independent training seeds that solve.
    # Deterministic rollouts are not treated as independent observations.
    z=1.959963984540054; n=summary.seeds.astype(float); p=summary.success_mean.astype(float)
    den=1+z*z/n; center=(p+z*z/(2*n))/den
    half=z*np.sqrt((p*(1-p)/n)+(z*z/(4*n*n)))/den
    summary["success_ci95_low"]=(center-half).clip(0,1)
    summary["success_ci95_high"]=(center+half).clip(0,1)
    summary.to_csv(out/"gamma_sweep_aggregate.csv",index=False)
    return rows,summary


if __name__ == "__main__": run()
