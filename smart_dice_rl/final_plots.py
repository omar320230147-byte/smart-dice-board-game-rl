"""Generate analysis figures directly from persisted experimental CSVs."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent.parent

def plot_gamma():
    p=ROOT/'results'/'improved'/'gamma_sweep'; d=pd.read_csv(p/'gamma_sweep_summary.csv')
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    for algo,g in d.groupby('algorithm'):
        a=g.groupby('gamma').success_rate.agg(['mean','std']).sort_index()
        ax.errorbar(a.index,a['mean'],yerr=a['std'],marker='o',capsize=3,label=algo.replace('_',' ').title())
    ax.set(title='Board 3 success by discount factor',xlabel='Discount factor γ',ylabel='Final success rate (mean ± SD across training seeds)',ylim=(-.05,1.05)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(p/'gamma_success.png',dpi=180); plt.close(fig)
    curves=pd.read_csv(p/'gamma_sweep_progress.csv')
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    for (alg,gamma),grp in curves.groupby(['algorithm','gamma']):
        vals=grp.groupby('episode').eval_success_rate.agg(['mean','std']).sort_index()
        ax.plot(vals.index,vals['mean'],label=f"{alg.upper()} γ={gamma:g}")
    ax.set(title='Discount sweep held-out learning checkpoints',xlabel='Training episode',ylabel='Success rate at separate checkpoint evaluation',ylim=(-.05,1.05)); ax.grid(alpha=.25); ax.legend(ncol=2,fontsize=8); fig.tight_layout(); fig.savefig(p/'gamma_learning.png',dpi=180); plt.close(fig)

def plot_reward():
    p=ROOT/'results'/'improved'/'reward_experiments'; d=pd.read_csv(p/'reward_summary.csv')
    g=d.groupby(['reward_mode','algorithm']).success_rate.agg(['mean','std']).reset_index()
    modes=list(g.reward_mode.unique()); x=np.arange(len(modes)); width=.34
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for j,alg in enumerate(('q_learning','sarsa')):
        sub=g[g.algorithm==alg].set_index('reward_mode').reindex(modes)
        ax.bar(x+(j-.5)*width,sub['mean'],width,yerr=sub['std'],capsize=3,label=alg.replace('_',' ').title())
    ax.set_xticks(x,modes,rotation=15); ax.set(title='Reward variants on the 100-square board',xlabel='Reward configuration',ylabel='Held-out success rate (mean ± SD across three seeds)',ylim=(0,1.05)); ax.grid(axis='y',alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(p/'reward_success.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.4,4.8))
    for (mode,alg),grp in pd.read_csv(p/'reward_progress.csv').groupby(['reward_mode','algorithm']):
        vals=grp.groupby('episode').success_rate.mean()
        ax.plot(vals.index,vals.values,label=f'{mode} / {alg}')
    ax.set(title='Training success under alternative rewards',xlabel='Training episode',ylabel='Rolling training success rate',ylim=(-.05,1.05)); ax.grid(alpha=.25); ax.legend(fontsize=7,ncol=2); fig.tight_layout(); fig.savefig(p/'reward_training.png',dpi=180); plt.close(fig)

def plot_generalization():
    p=ROOT/'results'/'improved'/'generalization'; d=pd.read_csv(p/'generalization_aggregate.csv')
    sub=d[(d.board_group=='test')]
    for alg,g in sub.groupby('algorithm'):
        fig,ax=plt.subplots(figsize=(7.6,4.6))
        for mode,gg in g.groupby('training_mode'):
            gg=gg.sort_values('board_size'); ax.errorbar(gg.board_size,gg.success_mean,yerr=gg.success_sd.fillna(0),marker='o',capsize=3,label=mode.replace('_',' ').title())
        ax.set(title=f'{alg}: held-out layout generalization',xlabel='Board squares',ylabel='Test-board success rate (mean ± SD)',xticks=[30,50,100],ylim=(-.05,1.05)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(p/f'heldout_{alg.lower()}.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    g=d[d.board_group=='test'].groupby(['training_mode','algorithm']).time_mean.mean().unstack('training_mode')
    g.plot(kind='bar',ax=ax); ax.set(title='Training wall time by regime and algorithm',xlabel='Algorithm',ylabel='Mean training seconds per seed'); ax.grid(axis='y',alpha=.25); ax.legend(title='Training regime'); fig.tight_layout(); fig.savefig(p/'training_time.png',dpi=180); plt.close(fig)
    bfs=d[d.board_group=='test'].groupby(['training_mode','board_size']).agg(agent_moves=('moves_mean','mean'),bfs=('bfs_min_moves','mean')).reset_index()
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    for mode,g in bfs.groupby('training_mode'):
        ax.plot(g.board_size,g.agent_moves,marker='o',label=f'{mode.replace("_"," ")} agents')
    ax.plot(bfs.drop_duplicates('board_size').board_size,bfs.drop_duplicates('board_size').bfs,marker='s',linestyle='--',color='black',label='BFS minimum')
    ax.set(title='Held-out RL episode length and BFS lower bound',xlabel='Board squares',ylabel='Moves (failures counted at step cap)',xticks=[30,50,100]);ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig(p/'bfs_test_comparison.png',dpi=180);plt.close(fig)

def plot_drl_tuning():
    p=ROOT/'results'/'improved'/'drl_experiments';d=pd.read_csv(p/'drl_tuning_aggregate.csv')
    sub=d[d.board_group=='test']; algos=['DQN','PPO','A2C']; variants=list(sub.variant.unique()); x=np.arange(len(algos));w=.35
    fig,ax=plt.subplots(figsize=(8.2,4.8))
    for j,v in enumerate(variants):
        g=sub[sub.variant==v].set_index('algorithm').reindex(algos)
        ax.bar(x+(j-.5)*w,g.success_mean,w,yerr=g.success_sd,capsize=3,label=v.replace('_',' '))
    ax.set_xticks(x,algos);ax.set(title='DRL tuning held-out test performance',xlabel='Algorithm',ylabel='Test success across three unseen layouts',ylim=(0,1.05));ax.grid(axis='y',alpha=.25);ax.legend();fig.tight_layout();fig.savefig(p/'drl_tuning_test.png',dpi=180);plt.close(fig)

def plot_gamma_all_boards():
    p=ROOT/'results'/'improved'/'gamma_sweep_all_boards';d=pd.read_csv(p/'gamma_sweep_all_boards_aggregate.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,(size,g) in zip(axes,d.groupby('board_size')):
        for algo,gg in g.groupby('algorithm'):
            ax.errorbar(gg.gamma,gg.success_mean,yerr=gg.success_sd,marker='o',capsize=3,label=algo.replace('_',' ').title())
        ax.set(title=f'{size} squares',xlabel='Gamma',xticks=[.8,.9,.95,.99],ylim=(-.05,1.05));ax.grid(alpha=.25)
    axes[0].set_ylabel('Success rate across training seeds');axes[-1].legend(fontsize=8)
    fig.suptitle('Ten-seed discount sweep across all required boards');fig.tight_layout();fig.savefig(p/'gamma_all_boards.png',dpi=180);plt.close(fig)

def run():
    plot_gamma(); plot_reward(); plot_generalization(); plot_drl_tuning(); plot_gamma_all_boards()

if __name__=='__main__': run()
