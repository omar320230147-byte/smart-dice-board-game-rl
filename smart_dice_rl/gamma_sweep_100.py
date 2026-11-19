"""Separate tabular discount sweep to diagnose positive-reward looping."""
from pathlib import Path
import time
import numpy as np,pandas as pd
from .improve_100 import ROOT,OUT,MODELS,load_board,EVAL_EPISODES
from .environment import SmartDiceEnv
from .tabular import TabularAgent
from .experiments import evaluate

SEEDS=(71,97); GAMMAS=(.80,.90); EPISODES=20000

def run():
    b=load_board(); dst=MODELS/'gamma_sweep'; dst.mkdir(parents=True,exist_ok=True); results=[]; curves=[]
    for gamma in GAMMAS:
        for seed in SEEDS:
            for algo in ('q_learning','sarsa'):
                env=SmartDiceEnv(b['size'],b['snakes'],b['ladders']); agent=TabularAgent(b['size'],algo,alpha=.1,gamma=gamma,epsilon=1.,epsilon_min=.02,epsilon_decay=.9998)
                def check(ep,a,s=seed):
                    metrics=evaluate(a,b,episodes=100,seed=810000+s,kind='tabular')
                    return {'eval_success_rate':metrics['success_rate'],'eval_avg_moves':metrics['avg_moves']}
                start=time.perf_counter()
                rows=agent.train(env,EPISODES,seed,progress_every=100,on_checkpoint=check)
                elapsed=time.perf_counter()-start; env.close()
                p=dst/f'board_3_100_{algo}_gamma{gamma:.2f}_seed{seed}.npz'; np.savez_compressed(p,q_table=agent.q)
                ev=evaluate(agent,b,episodes=EVAL_EPISODES,seed=990000+seed,kind='tabular')
                results.append({'board':'board_3_100','algorithm':algo,'gamma':gamma,'seed':seed,'training_episodes':EPISODES,'training_time_seconds':elapsed,'model_path':str(p.relative_to(ROOT)),**ev,'final_threshold_sustained':ev['success_rate']>=.95})
                curves.extend({'algorithm':algo,'gamma':gamma,'seed':seed,**row} for row in rows)
                print(f'gamma={gamma:.2f} {algo} seed={seed} win={ev["success_rate"]:.3f}, moves={ev["avg_moves"]:.1f}, reward={ev["avg_reward"]:.1f}',flush=True)
    (OUT/'csv').mkdir(parents=True,exist_ok=True)
    pd.DataFrame(results).to_csv(OUT/'csv'/'gamma_sweep_100_summary.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'csv'/'gamma_sweep_100_progress.csv',index=False)

if __name__=='__main__':run()
