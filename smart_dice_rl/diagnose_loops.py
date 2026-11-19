from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .environment import SmartDiceEnv
from .tabular import TabularAgent
from .sb3_agents import ALGORITHMS

ROOT=Path(__file__).resolve().parent.parent
rows=[]

def trace(agent,kind,board,tag):
    env=SmartDiceEnv(board['size'],board['snakes'],board['ladders']); obs,_=env.reset(); state=int(obs[0]); seen={}; transitions=[]
    for step in range(env.max_steps):
        if kind=='table': action=agent.act(state,np.random.default_rng(0),False)
        else:
            action,_=agent.predict(obs,deterministic=True); action=int(np.asarray(action).item())
        obs,r,term,tr,info=env.step(action); nxt=int(obs[0]); transitions.append((state,action+1,nxt,r,info['event']))
        if nxt in seen and not term:
            cycle=transitions[seen[nxt]:]
            rows.append({'model':tag,'repeated_state':nxt,'move_of_repeat':step+1,'cycle_length':len(cycle),'cycle_reward':sum(x[3] for x in cycle),'cycle':repr(cycle)})
            print(tag,'repeated state',nxt,'at move',step+1,'cycle length',len(cycle),'cycle reward',sum(x[3] for x in cycle),'cycle:',cycle[:20])
            return
        seen[nxt]=len(transitions); state=nxt
        if term: print(tag,'won in',step+1); return
        if tr: break
    print(tag,'no state cycle detected; ended at limit')

def run():
    b=json.loads((ROOT/'data'/'generated_boards'/'board_3_100.json').read_text(encoding='utf-8')); b['snakes']={int(k):int(v) for k,v in b['snakes'].items()}; b['ladders']={int(k):int(v) for k,v in b['ladders'].items()}
    for name in ('q_learning','sarsa'):
        a=TabularAgent(100,name); a.q=np.load(ROOT/'models'/f'board_3_100_{name}_seed17.npz')['q_table']; trace(a,'table',b,'original_'+name)
        a.q=np.load(ROOT/'models'/'improved'/f'board_3_100_{name}_seed71.npz')['q_table']; trace(a,'table',b,'improved_'+name)
    for name in ('DQN','PPO','A2C'):
        for p in [ROOT/'models'/f'board_3_100_{name.lower()}_seed17.zip',ROOT/'models'/'improved'/f'board_3_100_{name.lower()}_seed71.zip']:
            trace(ALGORITHMS[name].load(str(p),device='cpu'),'sb3',b,str(p.relative_to(ROOT)))
    import pandas as pd
    out=ROOT/'results'/'audit'/'reward_loop_diagnosis.csv'; out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print('Saved loop audit to',out)

if __name__=='__main__': run()
