"""Rebuild and independently test the best validation checkpoint from each tabular run."""
from pathlib import Path
import time
import numpy as np,pandas as pd
from .improve_100 import ROOT,OUT,MODELS,load_board,EVAL_EPISODES
from .environment import SmartDiceEnv
from .tabular import TabularAgent
from .experiments import evaluate

CHECKPOINTS={('q_learning',71):None,('q_learning',83):None,('q_learning',97):500,('sarsa',71):1500,('sarsa',83):500,('sarsa',97):1000}

def run():
    b=load_board(); outdir=MODELS/'checkpoint_selected'; outdir.mkdir(parents=True,exist_ok=True); rows=[]
    for (algo,seed),episode in CHECKPOINTS.items():
        if episode is None: continue
        env=SmartDiceEnv(b['size'],b['snakes'],b['ladders']); agent=TabularAgent(b['size'],algo,alpha=.1,gamma=.99,epsilon=1.,epsilon_min=.02,epsilon_decay=.9998)
        t=time.perf_counter()
        agent.train(env,episode,seed,progress_every=100,on_checkpoint=lambda ep,a,s=seed:{'eval_success_rate':evaluate(a,b,episodes=100,seed=810000+s,kind='tabular')['success_rate']})
        env.close(); elapsed=time.perf_counter()-t
        p=outdir/f'board_3_100_{algo}_seed{seed}_episode{episode}.npz'; np.savez_compressed(p,q_table=agent.q)
        held=evaluate(agent,b,episodes=EVAL_EPISODES,seed=950000+seed,kind='tabular')
        validation=evaluate(agent,b,episodes=100,seed=810000+seed,kind='tabular')
        rows.append({'board':'board_3_100','algorithm':algo,'seed':seed,'checkpoint_episode':episode,'training_budget_to_checkpoint':episode,'training_time_seconds':elapsed,'validation_success_rate':validation['success_rate'],'model_path':str(p.relative_to(ROOT)),**held})
        print(f'{algo} seed={seed} checkpoint={episode}: validation={validation["success_rate"]:.3f}, heldout={held["success_rate"]:.3f}, moves={held["avg_moves"]:.1f}',flush=True)
    pd.DataFrame(rows).to_csv(OUT/'csv'/'improved_100_selected_checkpoints.csv',index=False)

if __name__=='__main__':run()
