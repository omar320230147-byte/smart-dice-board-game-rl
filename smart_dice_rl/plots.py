from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def make_plots(out: Path):
    csv, plots = out / "csv", out / "plots"
    data = pd.read_csv(csv / "training_progress.csv")
    for metric, ylabel, filename in (("avg_reward", "Mean episode reward", "reward_training.png"), ("avg_moves", "Mean moves per episode", "moves_training.png"), ("success_rate", "Success rate (tabular window / DRL evaluation)", "success_training.png")):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        for ax, board in zip(axes, sorted(data.board.unique())):
            for algo, group in data[data.board == board].groupby("algorithm"):
                summary = group.groupby("episode")[metric].agg(["mean", "std"]).reset_index()
                ax.plot(summary.episode, summary["mean"], label=algo)
                if summary["std"].notna().any(): ax.fill_between(summary.episode, summary["mean"]-summary["std"].fillna(0), summary["mean"]+summary["std"].fillna(0), alpha=.15)
            ax.set(title=board, xlabel="Training progress (episode or timestep evaluation)", ylabel=ylabel); ax.grid(alpha=.2)
            if metric == "success_rate": ax.set_ylim(0, 1)
            elif metric == "avg_moves": ax.set_ylim(bottom=0)
        axes[0].legend(fontsize=8); fig.savefig(plots / filename, dpi=160); plt.close(fig)
    summary = pd.read_csv(csv / "evaluation_summary.csv")
    for algorithms, filename, title in ((["DQN", "PPO", "A2C"], "drl_comparison.png", "DRL evaluation success"), (["q_learning", "sarsa"], "tabular_comparison.png", "Tabular evaluation success")):
        sub=summary[summary.algorithm.isin(algorithms)]
        pivot=sub.groupby(["board_size","algorithm"]).success_rate.agg(["mean","std"]).reset_index()
        fig,ax=plt.subplots(figsize=(8,5))
        for algo,g in pivot.groupby("algorithm"):
            ax.errorbar(g.board_size,g["mean"],yerr=g["std"].fillna(0),marker="o",capsize=4,label=algo)
        ax.set(title=title,xlabel="Board size (squares)",ylabel="Success rate",ylim=(-.03,1.03)); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plots/filename,dpi=160); plt.close(fig)
    comparison = summary.groupby(["board_size", "algorithm"]).agg(success_mean=("success_rate", "mean"), success_std=("success_rate", "std"), moves_mean=("avg_moves", "mean"), moves_std=("avg_moves", "std"), reward_mean=("avg_reward", "mean"), reward_std=("avg_reward", "std"), training_time_mean=("training_time_seconds", "mean")).reset_index()
    comparison.to_csv(csv / "algorithm_comparison.csv", index=False)
    bfs = pd.read_csv(csv / "bfs_baseline.csv")
    learned = summary[summary.algorithm != "Random"].groupby("board_size").avg_moves.mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(bfs.board_size, bfs.bfs_min_moves, marker="o", label="BFS shortest path")
    ax.plot(learned.index, learned.values, marker="s", label="Mean RL moves (failures at limit included)")
    ax.set(title="RL and BFS move comparison", xlabel="Board size (squares)", ylabel="Moves per game"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(plots / "bfs_comparison.png", dpi=160); plt.close(fig)

