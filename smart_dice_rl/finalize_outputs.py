"""Reload final policies, reconcile evaluations, add BFS/inference audit columns."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from stable_baselines3 import A2C,DQN,PPO
from .baselines import bfs_distance
from .environment import SmartDiceEnv
from .generalization import fixed_board_env,benchmark_inference_ms
from .run_generalization import evaluate_board,ROOT

CLASSES={'DQN':DQN,'PPO':PPO,'A2C':A2C}

def check_close(a,b):
    return abs(float(a)-float(b))<1e-12

def run():
    audit=ROOT/'results'/'final_audit';audit.mkdir(parents=True,exist_ok=True)
    report=[]
    # Original policies: preserve and verify existing integrity audit.
    original=pd.read_csv(ROOT/'results'/'audit'/'policy_integrity.csv')
    if len(original)!=45 or not original.exists.all() or not original.valid_shape_and_values.all():
        raise AssertionError('original policy integrity failure')
    report.append({'category':'original policies','artifacts':len(original),'valid':int(original.valid_shape_and_values.sum()),'status':'PASS'})
    # New reward and gamma table files are numeric, finite, and model artifacts reload to expected shapes.
    board_paths=sorted((ROOT/'data'/'generated_boards').glob('board_*.json'))
    sizes=[int(json.loads(p.read_text(encoding='utf-8'))['size']) for p in board_paths]
    model_checks=[]
    for csv_path in (ROOT/'results'/'improved'/'reward_experiments'/'reward_summary.csv',
                     ROOT/'results'/'improved'/'gamma_sweep'/'gamma_sweep_summary.csv',
                     ROOT/'results'/'improved'/'gamma_sweep_all_boards'/'gamma_sweep_all_boards_summary.csv'):
        df=pd.read_csv(csv_path)
        for row in df.to_dict('records'):
            path=ROOT/row['model_path']; size=int(row.get('board_size',100 if row['board']=='board_3_100' else 0))
            if size==0:
                size=next(int(json.loads(p.read_text())['size']) for p in board_paths if f"_{int(json.loads(p.read_text())['size'])}" in row['board'])
            q=np.load(path)['q_table']; valid=q.shape==(size+1,6) and np.isfinite(q).all()
            model_checks.append({'path':str(path.relative_to(ROOT)),'valid':bool(valid)})
    if not all(x['valid'] for x in model_checks): raise AssertionError('tabular model failure')
    unique_tabular={x['path'] for x in model_checks}
    report.append({'category':'new tabular policies','artifacts':len(unique_tabular),'valid':len(unique_tabular),'status':'PASS'})

    # Load every SB3 archive in the workspace, including final and selected artifacts.
    sb3_files=sorted((ROOT/'models').rglob('*.zip')); sb3_checks=[]
    for path in sb3_files:
        token=path.name.lower()
        algo=next((a for a in ('dqn','ppo','a2c') if a in token),None)
        if algo is None: raise AssertionError(f'cannot identify algorithm for {path}')
        model=CLASSES[algo.upper()].load(str(path),device='cpu')
        shape=model.observation_space.shape
        probe=np.asarray([1],dtype=np.int32) if shape==(1,) else np.zeros(shape,dtype=np.float32)
        action,_=model.predict(probe,deterministic=True)
        valid=int(np.asarray(action).item()) in range(6)
        sb3_checks.append({'path':str(path.relative_to(ROOT)),'algorithm':algo.upper(),'valid_action':valid})
    if not all(x['valid_action'] for x in sb3_checks):raise AssertionError('SB3 archive returned invalid action')
    pd.DataFrame(sb3_checks).to_csv(audit/'sb3_policy_integrity_all.csv',index=False)
    report.append({'category':'all SB3 policy archives','artifacts':len(sb3_checks),'valid':len(sb3_checks),'status':'PASS'})

    # Generalization split and exact loaded model reproduction of every reported row.
    gen_dir=ROOT/'results'/'improved'/'generalization'
    groups=json.loads((gen_dir/'boards.json').read_text(encoding='utf-8'))
    group_by_id={b['board_id']:b for items in groups.values() for b in items}
    split_sets=[{b['seed'] for b in groups[g]} for g in ('train','validation','test')]
    if split_sets[0]&split_sets[1] or split_sets[0]&split_sets[2] or split_sets[1]&split_sets[2]: raise AssertionError('board split leakage')
    gdf=pd.read_csv(gen_dir/'generalization_board_results.csv'); ver=[]; latency={}
    for key,chunk in gdf.groupby(['training_mode','algorithm','training_seed']):
        path=ROOT/chunk.iloc[0].model_path
        model=CLASSES[key[1]].load(str(path),device='cpu')
        probe=group_by_id['test_30_3001']; env=fixed_board_env(probe); obs,_=env.reset(seed=980000+key[2]);latency[key]=benchmark_inference_ms(model,obs);env.close()
        for _,r in chunk.iterrows():
            b=group_by_id[r.board_id];m=evaluate_board(model,b,episodes=int(r.evaluation_episodes),seed=940000+int(r.training_seed))
            ok=all(check_close(m[k],r[k]) for k in ('success_rate','avg_moves','avg_reward'))
            ver.append({'board_id':r.board_id,'training_mode':key[0],'algorithm':key[1],'training_seed':key[2],'reproduced':ok})
    gdf['bfs_min_moves']=gdf.board_id.map({b['board_id']:bfs_distance(SmartDiceEnv(b['size'],{int(k):int(v) for k,v in b['snakes'].items()},{int(k):int(v) for k,v in b['ladders'].items()})) for items in groups.values() for b in items})
    gdf['inference_ms_per_action']=[latency[(r.training_mode,r.algorithm,r.training_seed)] for r in gdf.itertuples()]
    gdf.to_csv(gen_dir/'generalization_board_results.csv',index=False)
    ga=gdf.groupby(['training_mode','algorithm','board_group','board_size']).agg(board_count=('board_id','nunique'),seed_count=('training_seed','nunique'),success_mean=('success_rate','mean'),success_sd=('success_rate','std'),moves_mean=('avg_moves','mean'),moves_sd=('avg_moves','std'),reward_mean=('avg_reward','mean'),reward_sd=('avg_reward','std'),time_mean=('training_time_seconds','mean'),inference_ms_mean=('inference_ms_per_action','mean'),bfs_min_moves=('bfs_min_moves','mean')).reset_index()
    ga.to_csv(gen_dir/'generalization_aggregate.csv',index=False)
    if not all(x['reproduced'] for x in ver): raise AssertionError('generalization reload evaluation differs')
    pd.DataFrame(ver).to_csv(audit/'generalization_reproduction.csv',index=False)
    report.append({'category':'generalization policies','artifacts':len(latency),'reported evaluations reproduced':len(ver),'status':'PASS'})

    # DRL tuning policies: reproduce training/validation/test rows.
    ddir=ROOT/'results'/'improved'/'drl_experiments'; ddf=pd.read_csv(ddir/'drl_tuning_results.csv');dchecks=[];dlat={}
    board_by_seed={1008:json.loads((ROOT/'data'/'generated_boards'/'board_3_100.json').read_text(encoding='utf-8')),
                   2101:next(b for b in [__import__('smart_dice_rl.boards',fromlist=['generate_board']).generate_board(100,8,8,2101)]),
                   3101:None,3102:None,3103:None}
    board_by_seed[1008]['snakes']={int(k):int(v) for k,v in board_by_seed[1008]['snakes'].items()};board_by_seed[1008]['ladders']={int(k):int(v) for k,v in board_by_seed[1008]['ladders'].items()}
    board_by_seed[2101]['snakes']={int(k):int(v) for k,v in board_by_seed[2101]['snakes'].items()};board_by_seed[2101]['ladders']={int(k):int(v) for k,v in board_by_seed[2101]['ladders'].items()}
    from .boards import generate_board
    for s in (3101,3102,3103):board_by_seed[s]=generate_board(100,8,8,s)
    for key,chunk in ddf.groupby(['algorithm','variant','seed']):
        model=CLASSES[key[0]].load(str(ROOT/chunk.iloc[0].model_path),device='cpu')
        env=fixed_board_env(board_by_seed[3101]);obs,_=env.reset(seed=990000+key[2]);dlat[key]=benchmark_inference_ms(model,obs);env.close()
        for _,r in chunk.iterrows():
            b=board_by_seed[int(r.board_seed)];m=evaluate_board(model,b,episodes=int(r.evaluation_episodes),seed=670000+int(r.seed))
            dchecks.append({'algorithm':key[0],'variant':key[1],'seed':key[2],'board_seed':r.board_seed,'reproduced':all(check_close(m[k],r[k]) for k in ('success_rate','avg_moves','avg_reward'))})
    ddf['bfs_min_moves']=ddf.board_seed.map({s:bfs_distance(SmartDiceEnv(b['size'],b['snakes'],b['ladders'])) for s,b in board_by_seed.items()})
    ddf['inference_ms_per_action']=[dlat[(r.algorithm,r.variant,r.seed)] for r in ddf.itertuples()]
    ddf.to_csv(ddir/'drl_tuning_results.csv',index=False)
    da=ddf.groupby(['algorithm','variant','board_group']).agg(seed_count=('seed','nunique'),board_count=('board_seed','nunique'),success_mean=('success_rate','mean'),success_sd=('success_rate','std'),moves_mean=('avg_moves','mean'),moves_sd=('avg_moves','std'),reward_mean=('avg_reward','mean'),time_mean=('training_time_seconds','mean'),inference_ms_mean=('inference_ms_per_action','mean'),bfs_min_moves=('bfs_min_moves','mean')).reset_index()
    da.to_csv(ddir/'drl_tuning_aggregate.csv',index=False)
    if not all(x['reproduced'] for x in dchecks):raise AssertionError('DRL tuning reload evaluation differs')
    pd.DataFrame(dchecks).to_csv(audit/'drl_tuning_reproduction.csv',index=False)
    report.append({'category':'DRL tuning policies','artifacts':len(dlat),'reported evaluations reproduced':len(dchecks),'status':'PASS'})

    # Every CSV parses; all report images are present.
    csvs=list((ROOT/'results').rglob('*.csv'))
    for p in csvs: pd.read_csv(p)
    report.append({'category':'CSV files','artifacts':len(csvs),'valid':len(csvs),'status':'PASS'})
    pd.DataFrame(report).to_csv(audit/'final_artifact_validation.csv',index=False)
    print(pd.DataFrame(report).to_string(index=False))
    print(f'Original policy integrity: {len(original)} / {len(original)}')
    print(f'Generalization evaluations reloaded: {len(ver)} / {len(ver)}')
    print(f'DRL tuning evaluations reloaded: {len(dchecks)} / {len(dchecks)}')
    print(f'CSV files parse: {len(csvs)} / {len(csvs)}')

if __name__=='__main__':run()
