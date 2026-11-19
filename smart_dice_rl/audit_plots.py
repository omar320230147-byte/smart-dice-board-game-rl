from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parent.parent
BASE=ROOT/'results'; OUT=BASE/'improved'; CSV=OUT/'csv'; PLOTS=OUT/'plots'

def run():
    PLOTS.mkdir(parents=True,exist_ok=True)
    original=pd.read_csv(BASE/'csv'/'evaluation_summary.csv')
    long=pd.read_csv(CSV/'improved_100_summary.csv')
    selected=pd.read_csv(CSV/'improved_100_selected_checkpoints.csv')
    gamma=pd.read_csv(CSV/'gamma_sweep_100_summary.csv')
    rows=[]
    def add(experiment,algo,frame,seed_count,budget,gamma_value):
        if frame.empty:return
        rows.append({'board':'board_3_100','experiment':experiment,'algorithm':algo,'seed_count':seed_count,'training_budget':budget,'gamma':gamma_value,'success_rate_mean':frame.success_rate.mean(),'success_rate_std':frame.success_rate.std(ddof=1),'avg_moves_mean':frame.avg_moves.mean(),'avg_moves_std':frame.avg_moves.std(ddof=1),'avg_reward_mean':frame.avg_reward.mean(),'avg_reward_std':frame.avg_reward.std(ddof=1),'training_time_mean_seconds':frame.training_time_seconds.mean() if 'training_time_seconds' in frame else None})
    for algo in ('q_learning','sarsa','DQN','PPO','A2C'):
        add('original final policy',algo,original[(original.board=='board_3_100')&(original.algorithm==algo)],3,'3,500 episodes' if algo in ('q_learning','sarsa') else '6,000 timesteps',.99)
        add('extended final policy, unchanged reward',algo,long[long.algorithm==algo],3,'20,000 episodes' if algo in ('q_learning','sarsa') else '25,000 timesteps',.99)
    for algo in ('q_learning','sarsa'):
        add('selected gamma=.99 checkpoint',algo,selected[selected.algorithm==algo],len(selected[selected.algorithm==algo]),'500-1,500 episodes',.99)
        g=gamma[gamma.algorithm==algo]
        add('gamma sweep final policy',algo,g,len(g),'20,000 episodes',None)
    ran=pd.read_csv(BASE/'audit'/'random_policy_10000_episode_audit.csv')
    for _,r in ran.iterrows():
        rows.append({'board':r.board,'experiment':'10,000-episode random audit','algorithm':'Random','seed_count':1,'training_budget':0,'gamma':None,'success_rate_mean':r.success_rate,'success_rate_std':None,'avg_moves_mean':r.avg_moves,'avg_moves_std':None,'avg_reward_mean':r.avg_reward,'avg_reward_std':None,'training_time_mean_seconds':0})
    pd.DataFrame(rows).to_csv(CSV/'improved_comparison.csv',index=False)

    # Success-rate comparison on the same 100-square layout.
    algos=['q_learning','sarsa','DQN','PPO','A2C']; x=range(len(algos)); width=.23
    org=[]; ext=[]; tuned=[]
    for a in algos:
        o=original[(original.board=='board_3_100')&(original.algorithm==a)].success_rate
        e=long[long.algorithm==a].success_rate
        t=gamma[gamma.algorithm==a].success_rate if a in ('q_learning','sarsa') else pd.Series(dtype=float)
        org.append(o.mean() if len(o) else 0); ext.append(e.mean() if len(e) else 0); tuned.append(t.mean() if len(t) else 0)
    fig,ax=plt.subplots(figsize=(10,5))
    ax.bar([i-width for i in x],org,width,label='Original final (γ=.99)')
    ax.bar(x,ext,width,label='Longer budget final (γ=.99)')
    ax.bar([i+width for i in x],tuned,width,label='Gamma sweep final (γ=.80/.90; tabular only)')
    ax.set_xticks(list(x),['Q-Learning','SARSA','DQN','PPO','A2C']);ax.set_ylim(0,1.05);ax.set_ylabel('Held-out success rate');ax.set_title('100-square policies: original vs longer training and discount sweep');ax.grid(axis='y',alpha=.25);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(PLOTS/'original_vs_improved_success.png',dpi=170);plt.close(fig)

    # Moves retain failures at the 400-step cap; log scale keeps optimal routes visible.
    fig,ax=plt.subplots(figsize=(9,5))
    vals=[original[(original.board=='board_3_100')&(original.algorithm!='Random')].avg_moves.mean(),long.avg_moves.mean(),gamma.avg_moves.mean()]
    ax.bar(['Original final policies','Extended final γ=.99','Gamma sweep final γ=.80/.90'],vals,color=['#5277a3','#d27a2d','#48916d'])
    ax.axhline(9,color='black',linestyle='--',label='BFS optimum: 9 moves');ax.set_yscale('log');ax.set_ylabel('Mean moves (log scale; failures = 400)');ax.set_title('Mean moves on the same 100-square board');ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(PLOTS/'original_vs_improved_moves.png',dpi=170);plt.close(fig)

    # Discount sweep learning progress, with validation evaluated every 500 episodes.
    progress=pd.read_csv(CSV/'gamma_sweep_100_progress.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),sharey=True)
    for ax,algo in zip(axes,['q_learning','sarsa']):
        g=progress[(progress.algorithm==algo)&progress.eval_success_rate.notna()]
        for gamma_value,part in g.groupby('gamma'):
            m=part.groupby('episode').eval_success_rate.agg(['mean','std']).reset_index()
            ax.plot(m.episode,m['mean'],marker='o',label=f'γ={gamma_value:.2f}')
            ax.fill_between(m.episode,m['mean']-m['std'].fillna(0),m['mean']+m['std'].fillna(0),alpha=.15)
        ax.set_title(algo.replace('_',' ').title());ax.set_xlabel('Training episodes');ax.grid(alpha=.25);ax.legend()
    axes[0].set_ylabel('Separate validation win rate');fig.suptitle('Tabular discount-factor sweep on board 3');fig.tight_layout();fig.savefig(PLOTS/'gamma_sweep_learning.png',dpi=170);plt.close(fig)

if __name__=='__main__':run()
