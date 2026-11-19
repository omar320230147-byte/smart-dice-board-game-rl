"""Longer-budget, separately stored improvement experiments for the 100-square board."""
from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np,pandas as pd,torch
from stable_baselines3 import DQN,PPO,A2C
from stable_baselines3.common.monitor import Monitor
from .environment import SmartDiceEnv
from .tabular import TabularAgent
from .sb3_agents import PeriodicEvalCallback
from .experiments import evaluate

ROOT=Path(__file__).resolve().parent.parent
BOARD_PATH=ROOT/'data'/'generated_boards'/'board_3_100.json'
OUT=ROOT/'results'/'improved'; MODELS=ROOT/'models'/'improved'
SEEDS=(71,83,97); TAB_EPISODES=20000; SB3_STEPS=25000; EVAL_EPISODES=500

def load_board():
    b=json.loads(BOARD_PATH.read_text(encoding='utf-8'))
    b['snakes']={int(k):int(v) for k,v in b['snakes'].items()}; b['ladders']={int(k):int(v) for k,v in b['ladders'].items()}
    return b

def make_env(b): return SmartDiceEnv(b['size'],b['snakes'],b['ladders'])

def new_model(name,env,seed):
    common=dict(policy='MlpPolicy',env=env,seed=seed,device='cpu',policy_kwargs={'net_arch':[64,64]},verbose=0)
    if name=='DQN': return DQN(**common,learning_rate=5e-4,buffer_size=50000,learning_starts=1000,batch_size=64,gamma=.99,train_freq=4,target_update_interval=1000,exploration_fraction=.65,exploration_initial_eps=1.,exploration_final_eps=.05)
    if name=='PPO': return PPO(**common,learning_rate=2.5e-4,n_steps=512,batch_size=64,n_epochs=5,gamma=.99,gae_lambda=.95,ent_coef=.02)
    return A2C(**common,learning_rate=5e-4,n_steps=32,gamma=.99,gae_lambda=.95,ent_coef=.02)

def run():
    torch.manual_seed(0); OUT.mkdir(parents=True,exist_ok=True); (OUT/'csv').mkdir(exist_ok=True); (OUT/'plots').mkdir(exist_ok=True); MODELS.mkdir(parents=True,exist_ok=True)
    b=load_board(); board_id='board_3_100'; all_rows=[]; progress=[]
    (OUT/'config.json').write_text(json.dumps({'board':board_id,'seeds':SEEDS,'tabular_episodes':TAB_EPISODES,'sb3_timesteps':SB3_STEPS,'evaluation_episodes':EVAL_EPISODES,'tabular_hyperparameters':{'alpha':.1,'gamma':.99,'epsilon_start':1.0,'epsilon_min':.02,'epsilon_decay':.9998},'DQN':{'learning_rate':.0005,'buffer_size':50000,'learning_starts':1000,'batch_size':64,'exploration_fraction':.65,'final_epsilon':.05},'PPO':{'learning_rate':.00025,'n_steps':512,'batch_size':64,'n_epochs':5,'entropy':.02},'A2C':{'learning_rate':.0005,'n_steps':32,'entropy':.02}},indent=2),encoding='utf-8')
    for seed in SEEDS:
        for name in ('q_learning','sarsa'):
            env=make_env(b); agent=TabularAgent(b['size'],name,alpha=.1,gamma=.99,epsilon=1.,epsilon_min=.02,epsilon_decay=.9998); started=time.perf_counter()
            def checkpoint(ep,a,s=seed):
                m=evaluate(a,b,episodes=100,seed=810000+s,kind='tabular')
                return {'eval_success_rate':m['success_rate'],'eval_avg_moves':m['avg_moves'],'eval_avg_reward':m['avg_reward']}
            rows=agent.train(env,TAB_EPISODES,seed,progress_every=100,on_checkpoint=checkpoint); elapsed=time.perf_counter()-started; env.close()
            p=MODELS/f'board_3_100_{name}_seed{seed}.npz'; np.savez_compressed(p,q_table=agent.q)
            ev=evaluate(agent,b,episodes=EVAL_EPISODES,seed=910000+seed,kind='tabular')
            eval_rows=[x for x in rows if 'eval_success_rate' in x]; first=next((eval_rows[i]['episode'] for i in range(2,len(eval_rows)) if all(x['eval_success_rate']>=.95 for x in eval_rows[i-2:i+1])),'Not converged within training budget')
            all_rows.append({'board':board_id,'algorithm':name,'seed':seed,'training_budget':TAB_EPISODES,'training_time_seconds':elapsed,'model_path':str(p.relative_to(ROOT)),**ev,'first_threshold_episode':first,'final_threshold_sustained':ev['success_rate']>=.95})
            progress.extend({'algorithm':name,'seed':seed,**r} for r in rows)
            print(f'improved {name} seed={seed} success={ev["success_rate"]:.3f}, moves={ev["avg_moves"]:.1f}, time={elapsed:.1f}s',flush=True)
    for seed in SEEDS:
        for name,cls in [('DQN',DQN),('PPO',PPO),('A2C',A2C)]:
            env=Monitor(make_env(b)); model=new_model(name,env,seed)
            cb=PeriodicEvalCallback(lambda:make_env(b),every=5000,episodes=100,output=OUT)
            started=time.perf_counter(); model.learn(total_timesteps=SB3_STEPS,callback=cb,progress_bar=False); elapsed=time.perf_counter()-started
            p=MODELS/f'board_3_100_{name.lower()}_seed{seed}'; model.save(str(p)); ev=evaluate(model,b,episodes=EVAL_EPISODES,seed=910000+seed,kind='sb3')
            first=next((x['episode'] for i,x in enumerate(cb.rows) if i>=2 and all(y['success_rate']>=.95 for y in cb.rows[i-2:i+1])),'Not converged within training budget')
            all_rows.append({'board':board_id,'algorithm':name,'seed':seed,'training_budget':SB3_STEPS,'training_time_seconds':elapsed,'model_path':str(p.relative_to(ROOT))+'.zip',**ev,'first_threshold_episode':first,'final_threshold_sustained':ev['success_rate']>=.95})
            progress.extend({'algorithm':name,'seed':seed,**r} for r in cb.rows)
            env.close(); print(f'improved {name} seed={seed} success={ev["success_rate"]:.3f}, moves={ev["avg_moves"]:.1f}, time={elapsed:.1f}s',flush=True)
    pd.DataFrame(all_rows).to_csv(OUT/'csv'/'improved_100_summary.csv',index=False)
    pd.DataFrame(progress).to_csv(OUT/'csv'/'improved_100_progress.csv',index=False)
    print(f'Completed improved experiment: {OUT}',flush=True)

if __name__=='__main__': run()
