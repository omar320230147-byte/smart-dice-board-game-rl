"""Fixed-layout versus multi-layout, cross-size generalization experiments."""
from __future__ import annotations

import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import A2C,DQN,PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .boards import generate_board
from .generalization import BoardCollectionEnv, fixed_board_env, benchmark_inference_ms
from .environment import SmartDiceEnv
from .baselines import bfs_distance

ROOT=Path(__file__).resolve().parent.parent
CONFIGS=((30,3,3),(50,5,5),(100,8,8))
TRAIN_SEEDS=(1001,1002,1003,1051,1052,1053,1101,1102,1103)
VAL_SEEDS=(2001,2051,2101)
TEST_SEEDS=(3001,3002,3003,3051,3052,3053,3101,3102,3103)
ALGORITHMS={"DQN":DQN,"PPO":PPO,"A2C":A2C}
TIMESTEPS=15000

def make_boards(seeds):
    out=[]; by_size={}
    for cfg in CONFIGS:
        size,ns,nl=cfg
        # The fixed seed tuples are grouped in contiguous blocks of three by board size.
        if seeds is TRAIN_SEEDS:
            i={30:0,50:3,100:6}[size]; sized=[generate_board(size,ns,nl,s) for s in seeds[i:i+3]]
        elif seeds is VAL_SEEDS:
            i={30:0,50:1,100:2}[size]; sized=[generate_board(size,ns,nl,seeds[i])]
        else:
            i={30:slice(0,3),50:slice(3,6),100:slice(6,9)}[size]; sized=[generate_board(size,ns,nl,s) for s in seeds[i]]
        by_size[size]=sized; out.extend(sized)
    return out

def model_for(name,env,seed):
    common=dict(policy="MlpPolicy",env=env,seed=seed,device="cpu",policy_kwargs={"net_arch":[128,128]},verbose=0)
    if name=="DQN": return DQN(**common,learning_rate=5e-4,buffer_size=30000,learning_starts=500,batch_size=64,gamma=.95,
                                   train_freq=4,target_update_interval=500,exploration_fraction=.7,exploration_initial_eps=1.,exploration_final_eps=.05)
    if name=="PPO": return PPO(**common,learning_rate=2.5e-4,n_steps=512,batch_size=64,n_epochs=5,gamma=.95,gae_lambda=.95,ent_coef=.02)
    return A2C(**common,learning_rate=5e-4,n_steps=32,gamma=.95,gae_lambda=.95,ent_coef=.02)

def evaluate_board(model,board,episodes=10,seed=1):
    env=fixed_board_env(board); wins=[]; moves=[]; rewards=[]
    for ep in range(episodes):
        obs,_=env.reset(seed=seed+ep); done=False; n=0; total=0.; win=False
        while not done:
            action,_=model.predict(obs,deterministic=True)
            obs,r,t,tr,info=env.step(int(np.asarray(action).item())); total+=r; n+=1; done=t or tr; win=bool(info["success"])
        wins.append(win); moves.append(n); rewards.append(total)
    env.close()
    shortest=bfs_distance(SmartDiceEnv(board['size'],{int(k):int(v) for k,v in board['snakes'].items()},
                                       {int(k):int(v) for k,v in board['ladders'].items()}))
    return {"success_rate":float(np.mean(wins)),"avg_moves":float(np.mean(moves)),"avg_reward":float(np.mean(rewards)),"bfs_min_moves":shortest}

class ValidationCallback(BaseCallback):
    def __init__(self,boards,every,episodes,best_path):
        super().__init__(verbose=0); self.boards=boards; self.every=every; self.episodes=episodes; self.best_path=best_path
        self.rows=[]; self.best=(-1.,float("inf"))
    def _on_step(self):
        if self.num_timesteps % self.every: return True
        scores=[evaluate_board(self.model,b,self.episodes,seed=920000+i*100) for i,b in enumerate(self.boards)]
        success=float(np.mean([x["success_rate"] for x in scores])); moves=float(np.mean([x["avg_moves"] for x in scores]))
        self.rows.append({"timesteps":self.num_timesteps,"validation_success_rate":success,"validation_avg_moves":moves})
        key=(success,-moves)
        if key>(self.best[0],-self.best[1]):
            self.best=(success,moves); self.model.save(str(self.best_path))
        return True

def run():
    train=make_boards(TRAIN_SEEDS); val=make_boards(VAL_SEEDS); test=make_boards(TEST_SEEDS)
    # Make stable IDs without relying on untrusted generator content.
    def label(group,b): return f"{group}_{int(b['size'])}_{b['seed']}"
    out=ROOT/"results"/"improved"/"generalization"; model_root=ROOT/"models"/"improved"/"generalization"
    out.mkdir(parents=True,exist_ok=True); model_root.mkdir(parents=True,exist_ok=True)
    sets={"train":train,"validation":val,"test":test}
    (out/"boards.json").write_text(json.dumps({g:[{**b,"snakes":{str(k):v for k,v in b["snakes"].items()},"ladders":{str(k):v for k,v in b["ladders"].items()},"board_id":label(g,b)} for b in bs] for g,bs in sets.items()},indent=2),encoding="utf-8")
    rows=[]; curves=[]; seeds=(421,431,443); started_all=time.perf_counter()
    for name in ("DQN","PPO","A2C"):
        for mode in ("fixed_100","multi_board"):
            for seed in seeds:
                train_envs=[b for b in train if b["size"]==100 and b["seed"]==1101] if mode=="fixed_100" else train
                env=Monitor(BoardCollectionEnv(train_envs)); model=model_for(name,env,seed)
                best_path=model_root/f"{mode}_{name.lower()}_seed{seed}_best"
                callback=ValidationCallback(val,every=5000,episodes=5,best_path=best_path)
                t0=time.perf_counter(); model.learn(total_timesteps=TIMESTEPS,callback=callback,progress_bar=False); duration=time.perf_counter()-t0
                final_path=model_root/f"{mode}_{name.lower()}_seed{seed}_final"; model.save(str(final_path)); env.close()
                # Use the best checkpoint selected strictly on validation boards.
                selected=ALGORITHMS[name].load(str(best_path),device="cpu") if best_path.with_suffix(".zip").exists() else model
                latency_env=fixed_board_env(test[0]); latency_obs,_=latency_env.reset(seed=950000+seed)
                latency_ms=benchmark_inference_ms(selected,latency_obs);latency_env.close()
                for group,boards in (("train",train),("validation",val),("test",test)):
                    for b in boards:
                        score=evaluate_board(selected,b,episodes=10,seed=940000+seed)
                        rows.append({"training_mode":mode,"algorithm":name,"training_seed":seed,"board_group":group,
                                     "board_id":label(group,b),"board_size":b["size"],"board_seed":b["seed"],
                                     "training_timesteps":TIMESTEPS,"training_time_seconds":duration,
                                     "validation_checkpoint_count":len(callback.rows),"model_path":str(best_path.relative_to(ROOT))+".zip",
                                     "evaluation_episodes":10,"inference_ms_per_action":latency_ms,**score})
                curves.extend({"training_mode":mode,"algorithm":name,"seed":seed,**c} for c in callback.rows)
                print(f"generalization {mode} {name} seed={seed} test success={np.mean([r['success_rate'] for r in rows if r['training_mode']==mode and r['algorithm']==name and r['training_seed']==seed and r['board_group']=='test']):.3f}",flush=True)
    pd.DataFrame(rows).to_csv(out/"generalization_board_results.csv",index=False)
    pd.DataFrame(curves).to_csv(out/"generalization_validation_progress.csv",index=False)
    agg=pd.DataFrame(rows).groupby(["training_mode","algorithm","board_group","board_size"]).agg(
        board_count=("board_id","nunique"),seed_count=("training_seed","nunique"),success_mean=("success_rate","mean"),success_sd=("success_rate","std"),
        moves_mean=("avg_moves","mean"),moves_sd=("avg_moves","std"),reward_mean=("avg_reward","mean"),reward_sd=("avg_reward","std"),time_mean=("training_time_seconds","mean")).reset_index()
    agg.to_csv(out/"generalization_aggregate.csv",index=False)
    (out/"config.json").write_text(json.dumps({"train_seeds":TRAIN_SEEDS,"validation_seeds":VAL_SEEDS,"heldout_test_seeds":TEST_SEEDS,
        "training_timesteps_per_run":TIMESTEPS,"training_seed_count":len(seeds),"evaluation_rollouts_per_board":10,
        "input":"202 features: normalized position and board size plus 100-bin snake/ladder maps",
        "model_selection":"best validation success, ties broken by lower validation move count",
        "total_wall_seconds":time.perf_counter()-started_all},indent=2),encoding="utf-8")
    return rows,agg

if __name__=="__main__": run()
