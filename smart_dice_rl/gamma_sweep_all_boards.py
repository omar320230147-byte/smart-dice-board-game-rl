"""Extend the ten-seed gamma study to all three original board configurations."""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from .boards import CONFIGS,generate_board
from .environment import SmartDiceEnv
from .experiments import evaluate
from .tabular import TabularAgent
from .gamma_sweep import GAMMAS,SEEDS,EPISODES,ROOT

def run():
    source=ROOT/'data'/'generated_boards'; out=ROOT/'results'/'improved'/'gamma_sweep_all_boards'
    models=ROOT/'models'/'improved'/'gamma_sweep_all_boards';out.mkdir(parents=True,exist_ok=True);models.mkdir(parents=True,exist_ok=True)
    allrows=[];allprogress=[]
    for bi,cfg in enumerate(CONFIGS,1):
        b=json.loads((source/f'board_{bi}_{cfg[0]}.json').read_text(encoding='utf-8'))
        b['snakes']={int(k):int(v) for k,v in b['snakes'].items()};b['ladders']={int(k):int(v) for k,v in b['ladders'].items()}
        board_id=f'board_{bi}_{b["size"]}'
        if b['size']==100:
            prior=pd.read_csv(ROOT/'results'/'improved'/'gamma_sweep'/'gamma_sweep_summary.csv')
            prior['board_size']=100;allrows.extend(prior.to_dict('records'))
            p=pd.read_csv(ROOT/'results'/'improved'/'gamma_sweep'/'gamma_sweep_progress.csv')
            p['board']=board_id;p['board_size']=100;allprogress.extend(p.to_dict('records'))
            continue
        for gamma in GAMMAS:
            for algo in ('q_learning','sarsa'):
                for seed in SEEDS:
                    env=SmartDiceEnv(b['size'],b['snakes'],b['ladders'])
                    agent=TabularAgent(b['size'],algo,alpha=.1,gamma=gamma,epsilon=1.,epsilon_min=.02,epsilon_decay=.9998)
                    def checkpoint(ep,a,seed=seed):
                        m=evaluate(a,b,episodes=25,seed=810000+seed,kind='tabular')
                        return {'eval_success_rate':m['success_rate'],'eval_avg_moves':m['avg_moves'],'eval_avg_reward':m['avg_reward']}
                    started=time.perf_counter();points=agent.train(env,EPISODES,seed,progress_every=500,on_checkpoint=checkpoint);elapsed=time.perf_counter()-started;env.close()
                    path=models/f'{board_id}_{algo}_gamma{gamma:.2f}_seed{seed}.npz';np.savez_compressed(path,q_table=agent.q)
                    ev=evaluate(agent,b,episodes=100,seed=910000+seed,kind='tabular')
                    cps=[x for x in points if 'eval_success_rate' in x]
                    conv=next((cps[i]['episode'] for i in range(2,len(cps)) if all(x['eval_success_rate']>=.95 for x in cps[i-2:i+1])),None)
                    row={'board':board_id,'board_size':b['size'],'algorithm':algo,'gamma':gamma,'seed':seed,'training_episodes':EPISODES,'training_time_seconds':elapsed,
                         'convergence_episode':conv if conv is not None else 'Not converged within training budget','final_threshold_sustained':ev['success_rate']>=.95,
                         'failure_rate':1-ev['success_rate'],'model_path':str(path.relative_to(ROOT)),**ev}
                    allrows.append(row);allprogress.extend({'board':board_id,'board_size':b['size'],'algorithm':algo,'gamma':gamma,'seed':seed,**x} for x in points)
                    print(f'{board_id} gamma={gamma:.2f} {algo} seed={seed}: success={ev["success_rate"]:.2f}, moves={ev["avg_moves"]:.1f}',flush=True)
    df=pd.DataFrame(allrows); prog=pd.DataFrame(allprogress)
    df.to_csv(out/'gamma_sweep_all_boards_summary.csv',index=False);prog.to_csv(out/'gamma_sweep_all_boards_progress.csv',index=False)
    grouped=df.groupby(['board','board_size','algorithm','gamma']).agg(seeds=('seed','count'),success_mean=('success_rate','mean'),success_sd=('success_rate','std'),moves_mean=('avg_moves','mean'),moves_sd=('avg_moves','std'),reward_mean=('avg_reward','mean'),reward_sd=('avg_reward','std'),time_mean=('training_time_seconds','mean'),failures=('failure_rate',lambda x:int((x>0).sum()))).reset_index()
    z=1.959963984540054;n=grouped.seeds.astype(float);p=grouped.success_mean.astype(float);den=1+z*z/n;c=(p+z*z/(2*n))/den;h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    grouped['success_ci95_low']=(c-h).clip(0,1);grouped['success_ci95_high']=(c+h).clip(0,1);grouped.to_csv(out/'gamma_sweep_all_boards_aggregate.csv',index=False)
    (out/'config.json').write_text(json.dumps({'boards':'three saved original configurations','gammas':GAMMAS,'seeds':SEEDS,'episodes':EPISODES,'alpha':.1,'epsilon_start':1.,'epsilon_min':.02,'epsilon_decay':.9998,'reward':'original','evaluation_episodes':100,'checkpoint_eval_episodes':25,'board_3_source':'separate results/improved/gamma_sweep 10-seed run'},indent=2),encoding='utf-8')
    return df,grouped
if __name__=='__main__':run()
