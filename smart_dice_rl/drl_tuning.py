"""Small validation-selected SB3 hyperparameter experiments on board 3."""
import json,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C,DQN,PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from .boards import generate_board
from .generalization import BoardCollectionEnv,fixed_board_env,benchmark_inference_ms
from .run_generalization import model_for,evaluate_board,ROOT
from .improve_100 import load_board

ALGOS={"DQN":DQN,"PPO":PPO,"A2C":A2C}
TIMESTEPS=10000
SEEDS=(521,523)

class SelectCallback(BaseCallback):
    def __init__(self,algo,validation_board,params,every=2500):
        super().__init__(verbose=0); self.algo=algo; self.board=validation_board; self.params=params; self.every=every; self.rows=[]
        self.best=(-1.,float('inf')); self.best_path=None
    def _on_step(self):
        if self.num_timesteps % self.every:return True
        m=evaluate_board(self.model,self.board,episodes=10,seed=660000)
        self.rows.append({"timesteps":self.num_timesteps,"validation_success_rate":m['success_rate'],"validation_avg_moves":m['avg_moves'],"validation_avg_reward":m['avg_reward']})
        key=(m['success_rate'],-m['avg_moves'])
        if key>(self.best[0],-self.best[1]):
            self.best=(m['success_rate'],m['avg_moves']); self.model.save(str(self.best_path))
        return True

def tuned_model(algo,env,seed,variant):
    common=dict(policy='MlpPolicy',env=env,seed=seed,device='cpu',policy_kwargs={'net_arch':[128,128]},verbose=0)
    if algo=='DQN':
        p=({"learning_rate":5e-4,"buffer_size":50000,"learning_starts":500,"batch_size":64,"train_freq":4,"target_update_interval":500,"exploration_fraction":.65,"exploration_final_eps":.05}
           if variant=='balanced' else {"learning_rate":3e-4,"buffer_size":100000,"learning_starts":500,"batch_size":64,"train_freq":4,"target_update_interval":500,"exploration_fraction":.9,"exploration_final_eps":.03})
        return DQN(**common,**p,gamma=.9)
    if algo=='PPO':
        p=({"learning_rate":2.5e-4,"n_steps":512,"batch_size":64,"n_epochs":5,"ent_coef":.02}
           if variant=='balanced' else {"learning_rate":1e-4,"n_steps":1024,"batch_size":128,"n_epochs":5,"ent_coef":.05})
        return PPO(**common,**p,gamma=.9,gae_lambda=.95)
    p=({"learning_rate":5e-4,"n_steps":32,"ent_coef":.02}
       if variant=='balanced' else {"learning_rate":2.5e-4,"n_steps":64,"ent_coef":.05})
    return A2C(**common,**p,gamma=.9,gae_lambda=.95)

def run():
    torch.manual_seed(0)
    original=load_board(); validation=generate_board(100,8,8,2101)
    test=[generate_board(100,8,8,s) for s in (3101,3102,3103)]
    out=ROOT/'results'/'improved'/'drl_experiments'; models=ROOT/'models'/'improved'/'drl_tuning'
    out.mkdir(parents=True,exist_ok=True);models.mkdir(parents=True,exist_ok=True)
    (out/'config.json').write_text(json.dumps({"train_board":"original board_3_100","validation_board_seed":2101,"test_board_seeds":[3101,3102,3103],"variants":["balanced","exploration_or_rollout"],"seeds":SEEDS,"timesteps":TIMESTEPS,"selected_on":"validation success, then move count","reward":"original"},indent=2),encoding='utf-8')
    rows=[]; curves=[]
    for algo in ('DQN','PPO','A2C'):
        for variant in ('balanced','exploration_or_rollout'):
            for seed in SEEDS:
                env=Monitor(BoardCollectionEnv([original])); model=tuned_model(algo,env,seed,variant)
                cb=SelectCallback(algo,validation,{})
                cb.best_path=models/f'{algo.lower()}_{variant}_seed{seed}_best'
                start=time.perf_counter(); model.learn(total_timesteps=TIMESTEPS,callback=cb,progress_bar=False); duration=time.perf_counter()-start
                final=models/f'{algo.lower()}_{variant}_seed{seed}_final'; model.save(str(final)); env.close()
                chosen=ALGOS[algo].load(str(cb.best_path),device='cpu') if cb.best_path.with_suffix('.zip').exists() else model
                test_env=fixed_board_env(test[0]);test_obs,_=test_env.reset(seed=690000+seed);latency=benchmark_inference_ms(chosen,test_obs);test_env.close()
                for group,boards in [('train',[original]),('validation',[validation]),('test',test)]:
                    for b in boards:
                        m=evaluate_board(chosen,b,episodes=20,seed=670000+seed)
                        rows.append({"algorithm":algo,"variant":variant,"seed":seed,"board_group":group,"board_size":100,"board_seed":b['seed'],
                                     "training_timesteps":TIMESTEPS,"training_time_seconds":duration,"model_path":str(cb.best_path.relative_to(ROOT))+'.zip',
                                     "validation_checkpoints":len(cb.rows),"evaluation_episodes":20,"inference_ms_per_action":latency,**m})
                curves.extend({"algorithm":algo,"variant":variant,"seed":seed,**c} for c in cb.rows)
                print(f"DRL tune {algo}/{variant} seed={seed}: val={cb.best[0]:.2f}; test={np.mean([r['success_rate'] for r in rows if r['algorithm']==algo and r['variant']==variant and r['seed']==seed and r['board_group']=='test']):.2f}",flush=True)
    pd.DataFrame(rows).to_csv(out/'drl_tuning_results.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'drl_tuning_validation_progress.csv',index=False)
    d=pd.DataFrame(rows); agg=d.groupby(['algorithm','variant','board_group']).agg(seed_count=('seed','nunique'),board_count=('board_seed','nunique'),success_mean=('success_rate','mean'),success_sd=('success_rate','std'),moves_mean=('avg_moves','mean'),moves_sd=('avg_moves','std'),reward_mean=('avg_reward','mean'),time_mean=('training_time_seconds','mean')).reset_index()
    agg.to_csv(out/'drl_tuning_aggregate.csv',index=False)
    return rows,agg

if __name__=='__main__':run()
