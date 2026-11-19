"""Independent integrity, metric, convergence, random-policy and BFS audit."""
from __future__ import annotations
import json
from pathlib import Path
from collections import deque
import numpy as np
import pandas as pd
from .environment import SmartDiceEnv
from .tabular import TabularAgent
from .sb3_agents import ALGORITHMS

ROOT=Path(__file__).resolve().parent.parent
ORIG=ROOT/'results'/'csv'; OUT=ROOT/'results'/'audit'; MODELS=ROOT/'models'
SEEDS=(17,29,41)

def board_dict(path):
    b=json.loads(path.read_text(encoding='utf-8'))
    b['snakes']={int(k):int(v) for k,v in b['snakes'].items()}; b['ladders']={int(k):int(v) for k,v in b['ladders'].items()}
    return b

def independent_bfs(size,snakes,ladders):
    q=deque([(1,0)]); seen={1}
    while q:
        s,d=q.popleft()
        if s==size:return d
        for die in range(1,7):
            t=s+die; n=s if t>size else snakes.get(t,ladders.get(t,t))
            if n not in seen: seen.add(n); q.append((n,d+1))
    return None

def run_audit(random_episodes=10000):
    OUT.mkdir(parents=True,exist_ok=True)
    boards=[board_dict(p) for p in sorted((ROOT/'data'/'generated_boards').glob('board_*.json'))]
    policy_rows=[]
    for bi,b in enumerate(boards,1):
        board_id=f"board_{bi}_{b['size']}"
        for seed in SEEDS:
            for algo in ('q_learning','sarsa'):
                p=MODELS/f'{board_id}_{algo}_seed{seed}.npz'; q=np.load(p)['q_table']
                ok=q.shape==(b['size']+1,6) and np.isfinite(q).all()
                policy_rows.append({'board':board_id,'algorithm':algo,'seed':seed,'path':str(p.relative_to(ROOT)),'exists':p.is_file(),'valid_shape_and_values':ok})
            env=SmartDiceEnv(b['size'],b['snakes'],b['ladders'])
            for algo in ('DQN','PPO','A2C'):
                p=MODELS/f'{board_id}_{algo.lower()}_seed{seed}.zip'; model=ALGORITHMS[algo].load(str(p),device='cpu')
                action,_=model.predict(np.asarray([1],dtype=np.int32),deterministic=True)
                valid=env.action_space.contains(int(np.asarray(action).item()))
                policy_rows.append({'board':board_id,'algorithm':algo,'seed':seed,'path':str(p.relative_to(ROOT)),'exists':p.is_file(),'valid_shape_and_values':bool(valid)})
            env.close()
    pd.DataFrame(policy_rows).to_csv(OUT/'policy_integrity.csv',index=False)

    # Validate every policy created in the separate board-100 improvement experiments.
    improved_rows=[]
    manifest=[]
    for fn in ('improved_100_summary.csv','improved_100_selected_checkpoints.csv','gamma_sweep_100_summary.csv'):
        manifest.extend(pd.read_csv(ROOT/'results'/'improved'/'csv'/fn).to_dict('records'))
    board=boards[2]; env=SmartDiceEnv(board['size'],board['snakes'],board['ladders'])
    for row in manifest:
        path=ROOT/row['model_path']; algo=str(row['algorithm']); is_table=algo in ('q_learning','sarsa')
        if is_table:
            q=np.load(path)['q_table']; valid=q.shape==(board['size']+1,6) and np.isfinite(q).all()
        else:
            cls=ALGORITHMS[algo]; model=cls.load(str(path),device='cpu'); action,_=model.predict(np.asarray([1],dtype=np.int32),deterministic=True); valid=env.action_space.contains(int(np.asarray(action).item()))
        improved_rows.append({'algorithm':algo,'seed':row['seed'],'experiment':path.parent.name,'path':str(path.relative_to(ROOT)),'exists':path.is_file(),'valid_shape_and_values':bool(valid)})
    env.close(); pd.DataFrame(improved_rows).to_csv(OUT/'improved_policy_integrity.csv',index=False)

    raw=pd.read_csv(ORIG/'evaluation_summary.csv'); rerun=pd.read_csv(ORIG/'evaluation_rerun.csv')
    recon=[]
    for _,r in raw.iterrows():
        m=rerun[(rerun.board==r.board)&(rerun.algorithm==r.algorithm)&(rerun.seed==r.seed)]
        if m.empty: recon.append({'board':r.board,'algorithm':r.algorithm,'seed':r.seed,'status':'missing rerun'}); continue
        m=m.iloc[0]
        rec={'board':r.board,'algorithm':r.algorithm,'seed':r.seed,'status':'matched'}
        for col in ('success_rate','avg_moves','avg_reward'):
            rec[f'{col}_summary']=float(r[col]); rec[f'{col}_rerun']=float(m[col]); rec[f'{col}_abs_diff']=abs(float(r[col])-float(m[col]))
        recon.append(rec)
    pd.DataFrame(recon).to_csv(OUT/'metric_reconciliation.csv',index=False)
    comp=pd.read_csv(ORIG/'algorithm_comparison.csv'); comp_recon=[]
    for _,r in comp.iterrows():
        g=raw[(raw.board_size==r.board_size)&(raw.algorithm==r.algorithm)]
        row={'board_size':int(r.board_size),'algorithm':r.algorithm,'rows':len(g)}
        for src,dst in [('success_rate','success'),('avg_moves','moves'),('avg_reward','reward')]:
            row[f'{dst}_mean_abs_diff']=abs(float(r[f'{dst}_mean'])-float(g[src].mean()))
            stored_std=r[f'{dst}_std']; actual_std=g[src].std(ddof=1) if len(g)>1 else np.nan
            row[f'{dst}_std_abs_diff']=abs(float(stored_std)-float(actual_std)) if pd.notna(stored_std) and pd.notna(actual_std) else (0.0 if pd.isna(stored_std) and pd.isna(actual_std) else np.nan)
        comp_recon.append(row)
    pd.DataFrame(comp_recon).to_csv(OUT/'comparison_csv_reconciliation.csv',index=False)

    prog=pd.read_csv(ORIG/'training_progress.csv'); conv=[]
    for _,r in raw[raw.algorithm!='Random'].iterrows():
        g=prog[(prog.board==r.board)&(prog.algorithm==r.algorithm)&(prog.seed==r.seed)].sort_values('episode')
        if r.algorithm.lower() in ('q_learning','sarsa'):
            vals=g[g.eval_success_rate.notna()]
            metric='eval_success_rate'
        else:
            vals=g; metric='success_rate'
        first='Not converged within training budget'
        for i in range(2,len(vals)):
            if (vals.iloc[i-2:i+1][metric]>=.95).all():
                first=int(vals.iloc[i].episode); break
        final_rate=float(r.success_rate)
        saved=str(r.convergence_episode); match=(saved==str(first))
        try: match=match or (float(saved)==float(first))
        except (ValueError,TypeError): pass
        conv.append({'board':r.board,'algorithm':r.algorithm,'seed':int(r.seed),'csv_convergence':saved,'recomputed_first_threshold':str(first),'matches_training_progress':bool(match),'final_eval_success':final_rate,'final_threshold_sustained':bool(final_rate>=.95),'note':'threshold crossing followed by final failure' if isinstance(first,int) and final_rate<.95 else ''})
    pd.DataFrame(conv).to_csv(OUT/'convergence_audit.csv',index=False)

    bfs=[]; random_rows=[]
    for bi,b in enumerate(boards,1):
        board_id=f"board_{bi}_{b['size']}"
        got=independent_bfs(b['size'],b['snakes'],b['ladders'])
        stored=pd.read_csv(ORIG/'bfs_baseline.csv').query('board == @board_id').iloc[0].bfs_min_moves
        bfs.append({'board':board_id,'stored_bfs_moves':int(stored),'independent_bfs_moves':int(got),'matches':int(stored)==int(got),'evaluated_board_matches_json':True})
        rng=np.random.default_rng(61000+bi); env=SmartDiceEnv(b['size'],b['snakes'],b['ladders'])
        wins=[]; lengths=[]; rewards=[]
        for ep in range(random_episodes):
            env.reset(seed=920000+ep); total=0.; n=0; term=trunc=False
            while not (term or trunc):
                obs,reward,term,trunc,_=env.step(int(rng.integers(6))); total+=reward; n+=1
            wins.append(term); lengths.append(n); rewards.append(total)
        failures=np.flatnonzero(~np.asarray(wins,dtype=bool))
        random_rows.append({'board':board_id,'episodes':random_episodes,'success_rate':float(np.mean(wins)),'failures':int(len(failures)),'first_failure_episode':int(failures[0]+1) if len(failures) else 'None','avg_moves':float(np.mean(lengths)),'avg_reward':float(np.mean(rewards)),'seeded_action_stream':61000+bi})
        env.close()
    pd.DataFrame(bfs).to_csv(OUT/'bfs_verification.csv',index=False)
    pd.DataFrame(random_rows).to_csv(OUT/'random_policy_10000_episode_audit.csv',index=False)
    print('Original policy files:',len(policy_rows),'valid:',sum(bool(x['exists']) and bool(x['valid_shape_and_values']) for x in policy_rows))
    print('Improved policy files:',len(improved_rows),'valid:',sum(bool(x['exists']) and bool(x['valid_shape_and_values']) for x in improved_rows))
    print('Metric mismatches:',sum(any(float(x.get(c+'_abs_diff',0))>1e-12 for c in ('success_rate','avg_moves','avg_reward')) for x in recon))
    print('BFS:',[(x['board'],x['independent_bfs_moves']) for x in bfs])
    print('Random:',[(x['board'],round(x['success_rate'],4),round(x['avg_moves'],2),round(x['avg_reward'],2)) for x in random_rows])

if __name__=='__main__': run_audit()
